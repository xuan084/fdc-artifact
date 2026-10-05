"""Tests for the lock-v11 addendum machinery (obd_v11_eval gate + frozen design, prereg_v11, v11_analysis rules,
v11_replica, run_v11 task registry).  No test reads an eval outcome: every eval-path test uses fakes or synthetic
rows, and the raw eval reader is monkeypatched to raise wherever the gate must refuse."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dsswm.envs import obd_v11_eval as OE
from dsswm.stats import prereg, prereg_v11
from dsswm.stats import v11_analysis as VA
from dsswm.stats import v11_replica as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_EVAL_PATHS = {str(Path(OE.DATA_FILES_V11[n]).resolve()) for n in ("obd_eval_labels", "obd_eval_outcome")}


@pytest.fixture(autouse=True)
def eval_file_guard(monkeypatch):
    """Suite-wide guard (external reviewer v11 lock r1 F2): any open / stat-free read of a real OBD eval file fails the test."""
    import builtins
    import io
    import os as _os
    real_open, real_os_open = builtins.open, _os.open

    def _chk(f):
        try:
            if str(Path(_os.fspath(f)).resolve()) in _EVAL_PATHS:
                raise AssertionError(f"test touched a real eval file: {f}")
        except TypeError:
            pass

    def g_open(f, *a, **k):
        _chk(f)
        return real_open(f, *a, **k)

    def g_os_open(f, *a, **k):
        _chk(f)
        return real_os_open(f, *a, **k)

    monkeypatch.setattr(builtins, "open", g_open)
    monkeypatch.setattr(io, "open", g_open)
    monkeypatch.setattr(_os, "open", g_os_open)
    monkeypatch.setattr(OE, "_read_eval_rows_raw", lambda: (_ for _ in ()).throw(
        AssertionError("raw eval reader reached in a test without an explicit sentinel")))
    yield


def test_guard_is_active():
    with pytest.raises(AssertionError):
        open(OE.DATA_FILES_V11["obd_eval_outcome"], "rb")
    with pytest.raises(AssertionError):
        OE.verify_eval_bytes({})


# --------------------------------------------------------------------------------------------- eval gate (fakes)
def test_eval_refused_without_task_or_lock(tmp_path):
    with pytest.raises(PermissionError):
        OE.OBDEnvV11Frozen("eval", log_path=tmp_path / "log.jsonl")                       # no task id / sha
    with pytest.raises(PermissionError):
        OE.OBDEnvV11Frozen("eval", eval_task_id="v11_obd_full", lock_sha256="0" * 64,
                           gate_path=tmp_path / "missing.json", log_path=tmp_path / "log.jsonl")
    if not prereg_v11.ADDENDUM_PATH.exists():
        ok, why = prereg_v11.addendum_gate("v11_obd_full")
        assert not ok
    assert not (tmp_path / "log.jsonl").exists()                                          # refusals never log


def test_draft_never_authorises():
    ok, why = prereg_v11.addendum_gate("v11_obd_full", path=prereg_v11.DRAFT_PATH)
    assert not ok
    with pytest.raises(RuntimeError):
        prereg_v11.check_addendum({"status": "draft"})


def _fake_lock(sha="a" * 64, layer=None):
    return {"sha256": sha, "eval_tasks": {"v11_obd_full": {"layer": layer or OE.LAYER}},
            "data_sha256": {n: {"sha256": f"h-{n}"} for n in OE.DATA_FILES_V11}}


def test_eval_gate_checks_registered_layer(monkeypatch):
    fake = _fake_lock()
    monkeypatch.setattr(prereg_v11, "addendum_gate", lambda tid, path=None: (True, fake))
    assert OE.eval_gate("v11_obd_full", OE.LAYER) is fake
    with pytest.raises(PermissionError):
        OE.eval_gate("v11_obd_full", "OBD-OTHER")
    with pytest.raises(PermissionError):
        OE.eval_gate("t2", OE.LAYER)
    monkeypatch.setattr(prereg_v11, "addendum_gate", lambda tid, path=None: (False, "no lock"))
    with pytest.raises(PermissionError):
        OE.eval_gate("v11_obd_full", OE.LAYER)


# --------------------------------------------------------------------------------------------- eval gate (REAL gate)
def _real_lock_file(tmp_path, monkeypatch, task_ok=True):
    """A canonical, LOCKED v11 addendum built from the draft, written to tmp (hence never committed).  Code / input
    drift checks are neutralised (the draft may predate the working tree); everything else is the real gate."""
    from dsswm.stats import prereg_v6
    d = json.loads(prereg_v11.DRAFT_PATH.read_text())
    d = {k: v for k, v in d.items() if k not in ("sha256", "sha256_draft")}
    d.update(status="locked", git_commit="0" * 40)
    if not task_ok:
        d["eval_tasks"] = {"other_task": d["eval_tasks"]["v11_obd_full"]}
    d["sha256"] = prereg.canonical_hash(d)
    f = tmp_path / "prereg_lock_v11_addendum.json"
    f.write_text(json.dumps(d))
    rc, ri = prereg_v6.v6_code_drift, prereg_v6.v6_input_drift
    v11 = d["version"]
    monkeypatch.setattr(prereg_v6, "v6_code_drift", lambda add, *a, **k: [] if add.get("version") == v11
                        else rc(add, *a, **k))
    monkeypatch.setattr(prereg_v6, "v6_input_drift", lambda add, *a, **k: [] if add.get("version") == v11
                        else ri(add, *a, **k))
    return f, d


def _sentinels(monkeypatch, log):
    calls = []
    real = OE.file_sha256

    def fh(p):
        if str(Path(p).resolve()) in _EVAL_PATHS:
            calls.append(("eval_hash", Path(p).name, log.exists()))
            return {"eval_labels.pkl": d_hash("obd_eval_labels"), "eval_outcome.npy": d_hash("obd_eval_outcome")}[
                Path(p).name]
        return real(p)
    prov = json.loads(Path(OE.DATA_FILES_V11["obd_provenance"]).read_text())["files"]

    def d_hash(n):
        return prov[OE._PROV_KEY[n]]
    monkeypatch.setattr(OE, "file_sha256", fh)
    monkeypatch.setattr(OE, "_read_eval_rows_raw", lambda: calls.append(("read", log.exists())) or "rows")
    return calls


def test_real_gate_refuses_uncommitted_lock_without_eval_access(tmp_path, monkeypatch):
    f, d = _real_lock_file(tmp_path, monkeypatch)
    log = tmp_path / "access.jsonl"
    calls = _sentinels(monkeypatch, log)
    ok, why = prereg_v11.addendum_gate("v11_obd_full", path=f)
    assert not ok and ("repository" in why or "committed" in why), why
    with pytest.raises(PermissionError):
        OE.read_eval_rows("v11_obd_full", OE.LAYER, d["sha256"], gate_path=f, log_path=log)
    assert calls == [] and not log.exists()


def test_real_gate_refuses_unknown_task_without_eval_access(tmp_path, monkeypatch):
    f, d = _real_lock_file(tmp_path, monkeypatch, task_ok=False)
    monkeypatch.setattr(prereg_v11, "lock_history_check", lambda p: "0" * 40)
    log = tmp_path / "access.jsonl"
    calls = _sentinels(monkeypatch, log)
    ok, why = prereg_v11.addendum_gate("v11_obd_full", path=f)
    assert not ok and "not an eval task" in why
    with pytest.raises(PermissionError):
        OE.read_eval_rows("v11_obd_full", OE.LAYER, d["sha256"], gate_path=f, log_path=log)
    assert calls == [] and not log.exists()


def test_real_gate_layer_sha_logging_and_order(tmp_path, monkeypatch):
    """History check neutralised (tmp file is not committed); everything else real.  Wrong layer, missing / wrong
    caller sha and an unwritable log refuse with no eval access; the authorised call logs (fsync) BEFORE the eval
    files are hashed and BEFORE any row is read."""
    f, d = _real_lock_file(tmp_path, monkeypatch)
    monkeypatch.setattr(prereg_v11, "lock_history_check", lambda p: "0" * 40)
    ok, info = prereg_v11.addendum_gate("v11_obd_full", path=f)
    assert ok, info
    log = tmp_path / "access.jsonl"
    calls = _sentinels(monkeypatch, log)
    for layer, sha in (("OBD-OTHER", d["sha256"]), (OE.LAYER, None), (OE.LAYER, "b" * 64), (OE.LAYER, 123)):
        with pytest.raises(PermissionError):
            OE.read_eval_rows("v11_obd_full", layer, sha, gate_path=f, log_path=log)
    with pytest.raises(PermissionError):
        OE.read_eval_rows("other_task", OE.LAYER, d["sha256"], gate_path=f, log_path=log)
    assert calls == [] and not log.exists()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):                                               # log cannot be written -> no read
        OE.read_eval_rows("v11_obd_full", OE.LAYER, d["sha256"], gate_path=f, log_path=blocker / "sub" / "log.jsonl")
    assert calls == []
    assert OE.read_eval_rows("v11_obd_full", OE.LAYER, d["sha256"], gate_path=f, log_path=log) == "rows"
    assert calls == [("eval_hash", "eval_labels.pkl", True), ("eval_hash", "eval_outcome.npy", True), ("read", True)]
    rec = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(rec) == 1 and rec[0]["v11_addendum_sha256"] == d["sha256"]


def test_runner_data_binding_never_touches_eval(monkeypatch):
    import run_v11 as RV
    seen = []
    real = OE.file_sha256
    monkeypatch.setattr(OE, "file_sha256", lambda p: seen.append(str(Path(p).resolve())) or real(p))
    a = RV.data_sha_for(None)
    d = json.loads(prereg_v11.DRAFT_PATH.read_text())
    assert RV.data_sha_for(d) == a                                       # identical binding on dev and eval
    assert not set(seen) & _EVAL_PATHS


def test_eval_path_on_synthetic_rows(monkeypatch, tmp_path):
    """Eval code path end to end on SYNTHETIC rows: gate (faked) -> read (faked) -> frozen map, items and problems
    (no re-segmentation, no eval-weight costs), fallback routing counted."""
    fz = OE.load_frozen()
    rng = np.random.default_rng(0)
    pairs = sorted(fz["pair_map"])
    n = 30_000
    pick = rng.integers(0, len(pairs), n)
    u0 = np.array([pairs[i].split("|")[0] for i in pick], dtype=object)
    u3 = np.array([pairs[i].split("|")[1] for i in pick], dtype=object)
    u3[:5] = "unseen-uf3"
    u0[5:7] = "unseen-uf0"
    item = rng.integers(0, 80, n)
    y = rng.integers(0, 2, n).astype(float)
    monkeypatch.setattr(OE, "read_eval_rows", lambda tid, layer, sha, gate_path=None, log_path=None, reason=None:
                        (u0, u3, item, y, np.arange(n)))
    env = OE.OBDEnvV11Frozen("eval", eval_task_id="v11_obd_full", lock_sha256="a" * 64)
    assert env.half == "eval" and env.N == n and env.tau_R == n and env.pool_sizes.sum() == n
    sp = OE.frozen_problems(fz)
    assert np.array_equal(env.problems.cost, sp.cost) and np.array_equal(env.problems.budgets, sp.budgets)
    assert env.fallback_rows == {"pair": 5, "uf0": 2}
    assert env.seg[5] == fz["fallback_segment"] == 0
    for i in range(5):
        assert env.seg[i] == fz["other_map"][f"{u0[i]}|other"]
    assert np.array_equal(env.arm, np.isin(item, fz["treat_items"]).astype(int))


# --------------------------------------------------------------------------------------------- frozen design (dev)
def test_frozen_design_reproduces_dev_feasibility_env():
    from dsswm.envs.obd_v11 import OBDEnv
    e = OE.OBDEnvV11Frozen("dev")
    r = OBDEnv("uf0x3", 2)
    assert np.array_equal(e.seg, r.seg) and np.array_equal(e.arm, r.arm)
    assert np.array_equal(e.problems.cost, r.problems.cost) and np.array_equal(e.problems.budgets, r.problems.budgets)
    assert e.fallback_rows == {"pair": 0, "uf0": 0} and e.S == 9 and e.A == 2
    fz = OE.load_frozen()
    assert len(fz["treat_items"]) == len(fz["control_items"]) == 40
    assert not set(fz["treat_items"]) & set(fz["control_items"])
    assert np.allclose(e.w, fz["dev_segment_shares"])
    assert fz["cost"] == [[0, max(1, int(np.rint(8 * 9 * w)))] for w in fz["dev_segment_shares"]]
    tot = sum(c[1] for c in fz["cost"])
    assert fz["budgets"] == [int(math.floor(b * tot + 1e-9)) for b in fz["budget_frac"]]
    assert len(e.checkpoints(VA.K_GRID)) == 40
    with pytest.raises(RuntimeError):
        OE.load_frozen(expect_sha256="0" * 64)


def test_freeze_refuses_overwrite():
    with pytest.raises(RuntimeError):
        OE.freeze_design()


def test_assign_rules():
    fz = {"pair_map": {"a|1": 2, "a|2": 1}, "other_map": {"a|other": 3}, "fallback_segment": 0,
          "treat_items": [1, 2], "control_items": [0, 3]}
    seg, nfp, nfu = OE.assign_segments(np.array(["a", "a", "a", "b"], dtype=object),
                                       np.array(["1", "2", "9", "1"], dtype=object), fz)
    assert seg.tolist() == [2, 1, 3, 0] and (nfp, nfu) == (1, 1)
    assert OE.assign_arms([0, 1, 2, 3], fz).tolist() == [0, 1, 1, 0]
    with pytest.raises(RuntimeError):
        OE.assign_arms([7], fz)


def test_prereg_v11_schema():
    with pytest.raises(RuntimeError):
        prereg_v11._schema({"version": 1}, require_commit=False)
    add = {k: {"x": 1} for k in prereg_v11.V11_REQUIRED_KEYS}
    add.update(thresh=0.82, eps=5e-4, data_sha256={n: {} for n in OE.DATA_FILES_V11},
               frozen_gates={"g.json": "h"}, input_sha256={"g.json": "h"}, git_commit="zz")
    prereg_v11._schema(add, require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v11._schema(add, require_commit=True)
    for k, bad in (("thresh", 1.2), ("thresh", float("nan")), ("eps", None), ("thresh", True)):
        with pytest.raises(RuntimeError):
            prereg_v11._schema(dict(add, **{k: bad}), require_commit=False)
    with pytest.raises(RuntimeError):                                     # gate file not bound as an input
        prereg_v11._schema(dict(add, input_sha256={"g.json": "other"}), require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v11._schema(dict(add, data_sha256={"obd_dev": {}}), require_commit=False)


def test_v10_input_drift_check(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    v10 = {"input_sha256": {"a.txt": prereg._sha_file(tmp_path / "a.txt")}}
    prereg_v11.v10_input_drift_check(v10, {}, ws_root=tmp_path)
    (tmp_path / "a.txt").write_text("y")
    with pytest.raises(RuntimeError):
        prereg_v11.v10_input_drift_check(v10, {}, ws_root=tmp_path)


def test_finalize_refuses_existing_and_non_draft(tmp_path):
    out = tmp_path / "lock.json"
    out.write_text("{}")
    with pytest.raises(RuntimeError):
        prereg_v11.finalize_addendum("0" * 40, out_path=out)
    d = tmp_path / "d.json"
    d.write_text(json.dumps({"status": "locked"}))
    with pytest.raises(RuntimeError):
        prereg_v11.finalize_addendum("0" * 40, draft_path=d, out_path=tmp_path / "new.json")


# --------------------------------------------------------------------------------------------- rules
def _row(m, s, e, n80, tau=1000, false=False):
    return {"method": m, "seed": s, "eps": e, "N80_pen": n80, "tau_R": tau, "fwer_event": false,
            "false_by_k80": False, "n80_lt_tau": n80 < tau, "reached_stop": True, "validity": "rigorous",
            "exhaustion_at_k80": {"all": 0.0, "ctrl": 0.0, "treat": 0.0}, "schedule_digest": "s",
            "arrival_digest": f"a{s}"}


def _rule_rows(succ_by_eps, seeds=VA.DEV_SEEDS, hc_better_at=()):
    rows = []
    for e in VA.EPS_GRID:
        k = succ_by_eps.get(e, 0)
        for i, s in enumerate(seeds):
            rows.append(_row(VA.RIVAL, s, e, 500 if i < k else 1000))
            for j, name in enumerate(VA.HC_NAMES):
                v = (400 if (e in hc_better_at and j == 2) else 900 + j) if i < k + 5 else 1000
                rows.append(_row(name, s, e, v))
    return rows


def test_hc_and_eps_rule():
    rows = _rule_rows({3e-4: 30, 5e-4: 41, 7.5e-4: 50}, hc_better_at=(5e-4,))
    hc = VA.select_hc(rows)
    assert hc[repr(5e-4)]["name"] == VA.HC_NAMES[2] and hc[repr(3e-4)]["name"] == VA.HC_NAMES[0]
    sel = VA.select_eps(rows, hc)
    assert sel["selected_eps"] == 5e-4 and sel["rival"] == "HC-WoR-DP[tuned]" and sel["successes"] == 46
    assert [t["qualifies"] for t in sel["trace"]] == [False, True]
    none = VA.select_eps(_rule_rows({}), VA.select_hc(_rule_rows({})))
    assert none["selected_eps"] is None and len(none["trace"]) == 5
    with pytest.raises(ValueError):                                          # the rule never reads FDC-DP
        VA.select_eps(rows + [_row(VA.PRIMARY, 950, 5e-4, 100)], hc)
    with pytest.raises(ValueError):
        VA.select_eps([r for r in rows if not (r["method"] == VA.RIVAL and r["seed"] == 950)], hc)


def test_rule_tie_goes_to_rect():
    rows = []
    for e in VA.EPS_GRID:
        for s in VA.DEV_SEEDS:
            rows.append(_row(VA.RIVAL, s, e, 500))
            rows += [_row(n, s, e, 500) for n in VA.HC_NAMES]
    sel = VA.select_eps(rows, VA.select_hc(rows))
    assert sel["selected_eps"] == 3e-4 and sel["rival"] == VA.RIVAL


def test_thresh_rule():
    assert VA.thresh_rule(0.7234) == 0.824
    assert VA.thresh_rule(0.72) == 0.82
    assert VA.thresh_rule(0.85) == 0.90 and VA.thresh_rule(0.99) == 0.90
    for bad in (float("nan"), 0, -1, float("inf")):
        with pytest.raises(ValueError):
            VA.thresh_rule(bad)


def _p(ub):
    return {"ub95_one_sided": ub, "geomean_ratio": ub * 0.97}


def test_decide():
    ok = VA.decide(_p(0.70), 0, {"status": "pass"}, 0.82)
    assert ok["verdict"] == "positive_result_achieved" and ok["failing_components"] == []
    b = VA.decide(_p(0.82), 0, {"status": "pass"}, 0.82)
    assert b["verdict"] == "positive_result_not_achieved" and b["sub_label"] == "faster_below_1"
    b = VA.decide(_p(1.1), 1, {"status": "fail"}, 0.82)
    assert b["failing_components"] == ["rival_superiority_failed", "validity_failure", "replica_fail"]
    assert b["sub_label"] == "not_faster"
    for args in ((_p(0.5), 0, {"status": None}, 0.82), (_p(-1.0), 0, {"status": "pass"}, 0.82),
                 (_p(0.5), True, {"status": "pass"}, 0.82), (_p(0.5), 0, {"status": "pass"}, 1.5),
                 (_p(0.5), 0, {"status": "pass"}, 1)):
        with pytest.raises(ValueError):
            VA.decide(*args)


def test_analyse_cell_synthetic():
    seeds = list(range(30))
    rows = []
    for m, base in {VA.PRIMARY: 300, VA.RIVAL: 600, VA.HC_TUNED: 700, VA.RECT_HG: 550}.items():
        rows += [_row(m, s, 5e-4, base * (1 + 0.01 * (s % 5))) for s in seeds]
    c = VA.analyse_cell(rows, seeds, 5e-4, B=2000)
    assert c["comparisons"][f"{VA.PRIMARY}/{VA.RIVAL}"]["geomean_ratio"] == pytest.approx(0.5)
    assert set(c["comparisons"]) == {f"{VA.PRIMARY}/{m}" for m in VA.METHODS[1:]} | \
        {f"{VA.RIVAL}/{m}" for m in VA.DESCRIPTIVE}
    assert c["primary_false_streams"] == 0


def test_registry_constants():
    import run_v10_posthoc_G as G
    assert VA.HC_GRID == G.HC_GRID and VA.RULE_MIN_SUCCESS == G.RULE_MIN_SUCCESS
    assert VA.EVAL_SEEDS == tuple(range(39000, 39200)) and VA.DEV_SEEDS == tuple(range(950, 1000))
    assert VA.EPS_GRID == (3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3) and VA.K_GRID == 40 and VA.N80_K == 12
    assert VA.METHODS == ("TU-FDC-DP(b)", "RECT-BF-DP-TU", "HC-WoR-DP[tuned]", "RECT-HG-DP")
    from dsswm.stats.v6_analysis import B_BOOT, BOOT_SEED
    assert B_BOOT == 10_000 and BOOT_SEED == 42
    from dsswm.envs.posthoc_access_v10 import POSTHOC_SEEDS, LOCK_EVAL_SEEDS
    used = set(POSTHOC_SEEDS) | set(LOCK_EVAL_SEEDS) | set(range(30000, 38000))
    assert not used & set(VA.EVAL_SEEDS)


# --------------------------------------------------------------------------------------------- replica
def test_replica_groups_and_checks():
    assert R.groups_for("v11_obd_full") == (VA.METHODS,)
    seeds = list(range(12))
    rows = [_row(m, s, 5e-4, 300) for m in VA.METHODS for s in seeds]
    rep = [r for r in rows if r["seed"] < 10]
    ch = R.compute_checks("t", False, rows, rep, VA.METHODS, (5e-4,), seeds)
    assert all(c["pass"] for c in ch)
    rows[0] = dict(rows[0], schedule_digest="other")
    ch = R.compute_checks("t", False, rows, rep, VA.METHODS, (5e-4,), seeds)
    assert not all(c["pass"] for c in ch)


# --------------------------------------------------------------------------------------------- runner
def test_runner_registry_and_code_files():
    import run_v11 as RV
    assert RV.TASKS["v11_obd_full"]["half"] == "eval" and RV.TASKS["v11_obd_full"]["seeds"] == list(VA.EVAL_SEEDS)
    assert RV.TASKS["v11_obd_full"]["methods"] == list(VA.METHODS)
    assert [t for t, T in RV.TASKS.items() if T["half"] == "eval"] == ["v11_obd_full"]
    for t, T in RV.TASKS.items():
        if T["half"] == "dev":
            assert T["seeds"] == list(VA.DEV_SEEDS)
        if T["role"] == "eps_rule":
            assert not any(m.startswith(("TU-FDC", "FDC")) for m in T["methods"])
    assert RV.STOP_FRAC == 1.0 and RV.DELTA == 0.05
    code = Path(RV.CODE)
    for f in RV.CODE_FILES + RV.LOCK_ONLY_FILES[1:]:
        assert (code / f).exists(), f


def test_code_files_cover_every_imported_dsswm_module():
    """Every dsswm module loaded by the runner's dev/eval path (env, methods, stats, gate, seal, analysis) in a clean
    interpreter is hash-bound in CODE_FILES."""
    import subprocess
    import run_v11 as RV
    code = Path(RV.CODE).resolve()
    prog = ("import sys, json, importlib; sys.path.insert(0, '.'); import run_v11 as RV\n"
            "for m in ('dsswm.envs.obd_v11_eval', 'dsswm.envs.obd_v11', 'dsswm.stats.prereg_v11', "
            "'dsswm.stats.v11_seal', 'dsswm.stats.v11_replica', 'dsswm.baselines.fdc_dp', 'dsswm.baselines.rect_dp', "
            "'dsswm.baselines.rect_hg_dp', 'dsswm.envs.seg_v10', 'dsswm.stats.v6_replica', 'dsswm.stats.v6_analysis'):\n"
            "    importlib.import_module(m)\n"
            "RV.make(RV.VA.RECT_HG, [1, 2]); RV.make(RV.VA.HC_NAMES[0], [1, 2])\n"
            "print(json.dumps(sorted(getattr(m, '__file__', '') or '' for n, m in sys.modules.items() "
            "if n.startswith('dsswm'))))")
    out = subprocess.run([sys.executable, "-c", prog], cwd=str(code), capture_output=True, text=True, check=True)
    loaded = {str(Path(f).resolve().relative_to(code)) for f in json.loads(out.stdout.strip().splitlines()[-1]) if f}
    loaded = {f for f in loaded if not f.endswith("__init__.py")}
    missing = sorted(loaded - set(RV.CODE_FILES))
    assert not missing, missing


def test_dev_stream_identity_k40():
    """One dev stream per method on the frozen dev env at the K = 40 grid: same schedule digest across methods, no
    false certificate for the rigorous methods at this seed."""
    import run_v11 as RV
    from dsswm.baselines.fdc_dp import DEFAULT_NODE_LIMIT, GRID_RATIO
    T = dict(RV.TASKS["v11_obd_pilot"])
    cfg = {VA.PRIMARY: {"node_limit": DEFAULT_NODE_LIMIT, "grid_ratio": GRID_RATIO},
           VA.HC_TUNED: {"c": 0.5, "tf": 0.4}}
    RV.init_env(T, 5e-4, cfg)
    out = [RV.job((m, 950, 5e-4, "c", None, "d")) for m in VA.METHODS]
    assert all(r["error"] is None for r in out), [r["error"] for r in out]
    assert len({r["schedule_digest"] for r in out}) == 1 and len({r["arrival_digest"] for r in out}) == 1
    assert all(r["K_eval"] == 40 for r in out)


def test_data_binding_uses_provenance_for_eval_files():
    h = OE.current_data_hashes()
    prov = json.loads(Path(OE.DATA_FILES_V11["obd_provenance"]).read_text())["files"]
    assert h["obd_eval_outcome"] == prov["eval_outcome.npy"] and h["obd_eval_labels"] == prov["eval_labels.pkl"]
    assert h["obd_dev"] == prov["dev.pkl"]
    assert set(OE.DATA_FILES_V11) == {"obd_dev", "obd_eval_labels", "obd_eval_outcome", "obd_provenance"}
