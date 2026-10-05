"""Tests for the lock-v12 addendum machinery (obd_v12_eval gate + frozen designs, prereg_v12, v12_analysis rules,
run_v12 task registry).  No test reads an eval file: a suite-wide guard fails any open of a real OBD eval file
(random/women, random/men and random/all), and the raw eval reader / byte verifier are replaced by sentinels."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dsswm.envs import obd_v12_eval as OE
from dsswm.stats import prereg, prereg_v12
from dsswm.stats import v12_analysis as VA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_ALL = "$DATA_DIR/open_bandit"
_EVAL_PATHS = {str(Path(OE.DATA_FILES_V12[n]).resolve()) for n in OE.EVAL_FILES} | \
    {str(Path(f"{_ALL}/{f}").resolve()) for f in ("eval_labels.pkl", "eval_outcome.npy")}


@pytest.fixture(autouse=True)
def eval_file_guard(monkeypatch):
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
    monkeypatch.setattr(OE, "_read_eval_rows_raw", lambda c: (_ for _ in ()).throw(
        AssertionError("raw eval reader reached in a test without an explicit sentinel")))
    yield


def test_guard_is_active():
    for n in OE.EVAL_FILES:
        with pytest.raises(AssertionError):
            open(OE.DATA_FILES_V12[n], "rb")
    for c in OE.CAMPAIGNS:
        with pytest.raises(AssertionError):
            OE._verify_eval_bytes(c, {})


def test_byte_verifier_is_private_and_docstring_order():
    """external reviewer v11 lock r2 P2-1 / P2-2: no public unauthenticated byte verifier; the overview states log-before-verify."""
    assert "verify_eval_bytes" not in OE.__all__ and "_verify_eval_bytes" not in OE.__all__
    assert not hasattr(OE, "verify_eval_bytes")
    assert "_read_eval_rows_raw" not in OE.__all__
    doc = OE.__doc__
    i_gate, i_log = doc.index("(2) the v12 gate"), doc.index("(5) one access-log line")
    i_hash, i_read = doc.index("(6) only then"), doc.index("(7) eval rows are deserialised")
    assert i_gate < i_log < i_hash < i_read


# --------------------------------------------------------------------------------------------- eval gate (fakes)
@pytest.mark.parametrize("c", OE.CAMPAIGNS)
def test_eval_refused_without_task_or_lock(tmp_path, c):
    with pytest.raises(PermissionError):
        OE.OBDEnvV12Frozen(c, "eval", log_path=tmp_path / "log.jsonl")
    with pytest.raises(PermissionError):
        OE.OBDEnvV12Frozen(c, "eval", eval_task_id=f"v12_{c}_full", lock_sha256="0" * 64,
                           gate_path=tmp_path / "missing.json", log_path=tmp_path / "log.jsonl")
    if not prereg_v12.ADDENDUM_PATH.exists():
        ok, _ = prereg_v12.addendum_gate(f"v12_{c}_full")
        assert not ok
    assert not (tmp_path / "log.jsonl").exists()


def test_draft_never_authorises():
    ok, _ = prereg_v12.addendum_gate("v12_women_full", path=prereg_v12.DRAFT_PATH)
    assert not ok
    with pytest.raises(RuntimeError):
        prereg_v12.check_addendum({"status": "draft"})


def _fake_lock(sha="a" * 64):
    return {"sha256": sha, "eval_tasks": {f"v12_{c}_full": {"layer": OE.LAYERS[c], "campaign": c}
                                          for c in OE.CAMPAIGNS},
            "data_sha256": {n: {"sha256": f"h-{n}"} for n in OE.DATA_FILES_V12}}


def test_eval_gate_checks_registered_layer_and_campaign(monkeypatch):
    fake = _fake_lock()
    monkeypatch.setattr(prereg_v12, "addendum_gate", lambda tid, path=None: (True, fake))
    assert OE.eval_gate("women", "v12_women_full", OE.LAYERS["women"]) is fake
    with pytest.raises(PermissionError):
        OE.eval_gate("women", "v12_women_full", OE.LAYERS["men"])            # wrong layer
    with pytest.raises(PermissionError):
        OE.eval_gate("men", "v12_women_full", OE.LAYERS["women"])            # wrong campaign
    with pytest.raises(PermissionError):
        OE.eval_gate("women", "other", OE.LAYERS["women"])
    monkeypatch.setattr(prereg_v12, "addendum_gate", lambda tid, path=None: (False, "no lock"))
    with pytest.raises(PermissionError):
        OE.eval_gate("women", "v12_women_full", OE.LAYERS["women"])


# --------------------------------------------------------------------------------------------- eval gate (REAL gate)
def _real_lock_file(tmp_path, monkeypatch, task_ok=True):
    """A canonical LOCKED v12 addendum built from the draft, written to tmp (never committed).  Code / input drift
    checks of v12 are neutralised (the draft may predate the working tree); everything else is the real gate."""
    from dsswm.stats import prereg_v6
    if not prereg_v12.DRAFT_PATH.exists():
        pytest.skip("v12 draft not built yet")
    d = json.loads(prereg_v12.DRAFT_PATH.read_text())
    d = {k: v for k, v in d.items() if k not in ("sha256", "sha256_draft")}
    d.update(status="locked", git_commit="0" * 40)
    if not task_ok:
        d["eval_tasks"] = {"other_task": d["eval_tasks"]["v12_women_full"]}
    d["sha256"] = prereg.canonical_hash(d)
    f = tmp_path / "prereg_lock_v12_addendum.json"
    f.write_text(json.dumps(d))
    rc, ri = prereg_v6.v6_code_drift, prereg_v6.v6_input_drift
    v12 = d["version"]
    monkeypatch.setattr(prereg_v6, "v6_code_drift", lambda add, *a, **k: [] if add.get("version") == v12
                        else rc(add, *a, **k))
    monkeypatch.setattr(prereg_v6, "v6_input_drift", lambda add, *a, **k: [] if add.get("version") == v12
                        else ri(add, *a, **k))
    return f, d


def _sentinels(monkeypatch, log):
    calls = []
    real = OE.file_sha256

    def fh(p):
        rp = str(Path(p).resolve())
        if rp in _EVAL_PATHS:
            c = Path(p).parent.name
            calls.append(("eval_hash", c, Path(p).name, log.exists()))
            return json.loads(Path(OE.DATA_FILES_V12[f"{c}_provenance"]).read_text())["files"][Path(p).name]
        return real(p)
    monkeypatch.setattr(OE, "file_sha256", fh)
    monkeypatch.setattr(OE, "_read_eval_rows_raw", lambda c: calls.append(("read", c, log.exists())) or "rows")
    return calls


def test_real_gate_refuses_uncommitted_lock_without_eval_access(tmp_path, monkeypatch):
    f, d = _real_lock_file(tmp_path, monkeypatch)
    log = tmp_path / "access.jsonl"
    calls = _sentinels(monkeypatch, log)
    ok, why = prereg_v12.addendum_gate("v12_women_full", path=f)
    assert not ok and ("repository" in why or "committed" in why), why
    with pytest.raises(PermissionError):
        OE.read_eval_rows("women", "v12_women_full", OE.LAYERS["women"], d["sha256"], gate_path=f, log_path=log)
    assert calls == [] and not log.exists()


def test_real_gate_refuses_unknown_task_without_eval_access(tmp_path, monkeypatch):
    f, d = _real_lock_file(tmp_path, monkeypatch, task_ok=False)
    monkeypatch.setattr(prereg_v12, "lock_history_check", lambda p: "0" * 40)
    log = tmp_path / "access.jsonl"
    calls = _sentinels(monkeypatch, log)
    ok, why = prereg_v12.addendum_gate("v12_women_full", path=f)
    assert not ok and "not an eval task" in why
    with pytest.raises(PermissionError):
        OE.read_eval_rows("women", "v12_women_full", OE.LAYERS["women"], d["sha256"], gate_path=f, log_path=log)
    assert calls == [] and not log.exists()


@pytest.mark.parametrize("c", OE.CAMPAIGNS)
def test_real_gate_layer_sha_logging_and_order(tmp_path, monkeypatch, c):
    f, d = _real_lock_file(tmp_path, monkeypatch)
    monkeypatch.setattr(prereg_v12, "lock_history_check", lambda p: "0" * 40)
    t = f"v12_{c}_full"
    o = "men" if c == "women" else "women"
    ok, info = prereg_v12.addendum_gate(t, path=f)
    assert ok, info
    log = tmp_path / "access.jsonl"
    calls = _sentinels(monkeypatch, log)
    for cc, layer, sha, task in ((c, OE.LAYERS[o], d["sha256"], t), (c, OE.LAYERS[c], None, t),
                                 (c, OE.LAYERS[c], "b" * 64, t), (c, OE.LAYERS[c], 123, t),
                                 (o, OE.LAYERS[o], d["sha256"], t), (c, OE.LAYERS[c], d["sha256"], "other"),
                                 ("all", OE.LAYERS[c], d["sha256"], t)):
        with pytest.raises(PermissionError):
            OE.read_eval_rows(cc, task, layer, sha, gate_path=f, log_path=log)
    assert calls == [] and not log.exists()
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        OE.read_eval_rows(c, t, OE.LAYERS[c], d["sha256"], gate_path=f, log_path=blocker / "sub" / "log.jsonl")
    assert calls == []
    assert OE.read_eval_rows(c, t, OE.LAYERS[c], d["sha256"], gate_path=f, log_path=log) == "rows"
    assert calls == [("eval_hash", c, "eval_labels.pkl", True), ("eval_hash", c, "eval_outcome.npy", True),
                     ("read", c, True)]
    rec = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(rec) == 1 and rec[0]["v12_addendum_sha256"] == d["sha256"] and rec[0]["campaign"] == c


def test_runner_data_binding_never_touches_eval(monkeypatch):
    import run_v12 as RV
    seen = []
    real = OE.file_sha256
    monkeypatch.setattr(OE, "file_sha256", lambda p: seen.append(str(Path(p).resolve())) or real(p))
    a = RV.data_sha_for(None)
    if prereg_v12.DRAFT_PATH.exists():
        d = json.loads(prereg_v12.DRAFT_PATH.read_text())
        assert RV.data_sha_for(d) == a
    assert not set(seen) & _EVAL_PATHS


def test_data_binding_uses_provenance_for_eval_files():
    h = OE.current_data_hashes()
    for c in OE.CAMPAIGNS:
        prov = json.loads(Path(OE.DATA_FILES_V12[f"{c}_provenance"]).read_text())["files"]
        assert h[f"{c}_eval_outcome"] == prov["eval_outcome.npy"] and h[f"{c}_eval_labels"] == prov["eval_labels.pkl"]
        assert h[f"{c}_dev"] == prov["dev.pkl"]
    assert len(OE.DATA_FILES_V12) == 8 and set(OE.EVAL_FILES) == {f"{c}_{k}" for c in OE.CAMPAIGNS
                                                                   for k in ("eval_labels", "eval_outcome")}


@pytest.mark.parametrize("c", OE.CAMPAIGNS)
def test_eval_path_on_synthetic_rows(monkeypatch, c):
    fz = OE.load_frozen(c)
    rng = np.random.default_rng(0)
    pairs = sorted(fz["pair_map"])
    n = 30_000
    pick = rng.integers(0, len(pairs), n)
    u0 = np.array([pairs[i].split("|")[0] for i in pick], dtype=object)
    u3 = np.array([pairs[i].split("|")[1] for i in pick], dtype=object)
    u3[:5] = "unseen-uf3"
    u0[5:7] = "unseen-uf0"
    item = rng.integers(0, OE.N_ITEMS[c], n)
    y = rng.integers(0, 2, n).astype(float)
    monkeypatch.setattr(OE, "read_eval_rows", lambda cc, tid, layer, sha, gate_path=None, log_path=None, reason=None:
                        (u0, u3, item, y, np.arange(n)))
    env = OE.OBDEnvV12Frozen(c, "eval", eval_task_id=f"v12_{c}_full", lock_sha256="a" * 64)
    assert env.half == "eval" and env.N == n and env.tau_R == n and env.pool_sizes.sum() == n
    sp = OE.frozen_problems(fz)
    assert np.array_equal(env.problems.cost, sp.cost) and np.array_equal(env.problems.budgets, sp.budgets)
    assert env.fallback_rows == {"pair": 5, "uf0": 2}
    assert env.seg[5] == fz["fallback_segment"] == 0
    for i in range(5):
        assert env.seg[i] == fz["other_map"][f"{u0[i]}|other"]
    assert np.array_equal(env.arm, np.isin(item, fz["treat_items"]).astype(int))
    with pytest.raises(RuntimeError):                                     # item outside the frozen lists
        OE.assign_arms([OE.N_ITEMS[c] + 5], fz)


# --------------------------------------------------------------------------------------------- frozen design (dev)
@pytest.mark.parametrize("c", OE.CAMPAIGNS)
def test_frozen_design_follows_v11_rules_on_dev(c):
    from dsswm.envs.obd_v11 import item_groups, segmentation
    df = OE.obd_dev_v12(c)
    e = OE.OBDEnvV12Frozen(c, "dev")
    ref, _ = segmentation(df, "uf0x3")
    grp = item_groups(df, 2)
    assert np.array_equal(e.seg, ref)
    assert np.array_equal(e.arm, df["item_id"].map(grp).to_numpy())
    fz = OE.load_frozen(c)
    h = OE.N_ITEMS[c] // 2
    assert len(fz["treat_items"]) == h and len(fz["control_items"]) == OE.N_ITEMS[c] - h
    assert not set(fz["treat_items"]) & set(fz["control_items"])
    ctr = df.groupby("item_id").click.mean()
    assert min(ctr[i] for i in fz["treat_items"]) >= max(ctr[i] for i in fz["control_items"])
    assert e.fallback_rows == {"pair": 0, "uf0": 0} and e.S == fz["S"] and e.A == 2
    assert np.allclose(e.w, fz["dev_segment_shares"])
    S = fz["S"]
    assert fz["cost"] == [[0, max(1, int(np.rint(8 * S * w)))] for w in fz["dev_segment_shares"]]
    tot = sum(x[1] for x in fz["cost"])
    assert fz["budgets"] == [int(math.floor(b * tot + 1e-9)) for b in fz["budget_frac"]]
    assert len(fz["budgets"]) == 15 and len(e.checkpoints(VA.K_GRID)) == 40
    with pytest.raises(RuntimeError):
        OE.load_frozen(c, expect_sha256="0" * 64)
    with pytest.raises(RuntimeError):
        OE.load_frozen("men" if c == "women" else "women", path=OE.frozen_path(c))


@pytest.mark.parametrize("c", OE.CAMPAIGNS)
def test_freeze_refuses_overwrite(c):
    with pytest.raises(RuntimeError):
        OE.freeze_design(c)


# --------------------------------------------------------------------------------------------- prereg_v12
def _schema_add():
    add = {k: {"x": 1} for k in prereg_v12.V12_REQUIRED_KEYS}
    add.update(blocks={c: {"eps": 5e-4, "thresh": 0.9, "status": "confirmatory", "layer": OE.LAYERS[c]}
                       for c in OE.CAMPAIGNS},
               eval_tasks={f"v12_{c}_full": {"campaign": c, "layer": OE.LAYERS[c], "eps": 5e-4} for c in OE.CAMPAIGNS},
               data_sha256={n: {} for n in OE.DATA_FILES_V12}, frozen_gates={"g.json": "h"},
               input_sha256={"g.json": "h"}, git_commit="zz")
    return add


def test_prereg_v12_schema():
    with pytest.raises(RuntimeError):
        prereg_v12._schema({"version": 1}, require_commit=False)
    add = _schema_add()
    prereg_v12._schema(add, require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v12._schema(add, require_commit=True)
    bad_blocks = [dict(add["blocks"], women=dict(add["blocks"]["women"], thresh=1.2)),
                  dict(add["blocks"], women=dict(add["blocks"]["women"], eps=float("nan"))),
                  dict(add["blocks"], women=dict(add["blocks"]["women"], status="maybe")),
                  dict(add["blocks"], women=dict(add["blocks"]["women"], layer=OE.LAYERS["men"])),
                  {"women": add["blocks"]["women"]}]
    for b in bad_blocks:
        with pytest.raises(RuntimeError):
            prereg_v12._schema(dict(add, blocks=b), require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v12._schema(dict(add, eval_tasks={"v12_women_full": {"campaign": "women", "layer": OE.LAYERS["women"],
                                                                     "eps": 1e-3}}), require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v12._schema(dict(add, eval_tasks={"t": {"campaign": "men", "layer": OE.LAYERS["women"],
                                                        "eps": 5e-4}}), require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v12._schema(dict(add, input_sha256={"g.json": "other"}), require_commit=False)
    with pytest.raises(RuntimeError):
        prereg_v12._schema(dict(add, data_sha256={"women_dev": {}}), require_commit=False)


def test_v11_input_drift_check(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    v11 = {"input_sha256": {"a.txt": prereg._sha_file(tmp_path / "a.txt")}}
    prereg_v12.v11_input_drift_check(v11, {}, ws_root=tmp_path)
    (tmp_path / "a.txt").write_text("y")
    with pytest.raises(RuntimeError):
        prereg_v12.v11_input_drift_check(v11, {}, ws_root=tmp_path)


def test_finalize_refuses_existing_and_non_draft(tmp_path):
    out = tmp_path / "lock.json"
    out.write_text("{}")
    with pytest.raises(RuntimeError):
        prereg_v12.finalize_addendum("0" * 40, out_path=out)
    d = tmp_path / "d.json"
    d.write_text(json.dumps({"status": "locked"}))
    with pytest.raises(RuntimeError):
        prereg_v12.finalize_addendum("0" * 40, draft_path=d, out_path=tmp_path / "new.json")


# --------------------------------------------------------------------------------------------- rules
def _share(mean, n_all_ctrl, n_q=15):
    return {"mean_share_eps_optimal": mean, "per_problem": [mean] * n_q, "n_problems_all_control_eps_optimal":
            n_all_ctrl, "n_policies": 256}


def test_block_status_gate():
    sel = {"selected_eps": 5e-4}
    assert VA.block_status(sel, _share(0.2, 0))["status"] == "confirmatory"
    assert VA.block_status(sel, _share(0.5, 14))["status"] == "confirmatory"          # boundaries inclusive
    assert VA.block_status(sel, _share(0.5001, 0))["status"] == "descriptive"
    assert VA.block_status(sel, _share(0.2, 15))["status"] == "descriptive"
    none = VA.block_status({"selected_eps": None}, _share(0.1, 0))
    assert none["status"] == "descriptive" and none["eps"] == 1.5e-3 == VA.FALLBACK_EPS
    assert none["criteria"]["i_rule_selects_eps"] is False
    assert VA.block_eps({"selected_eps": 7.5e-4}) == 7.5e-4


def _p(ub):
    return {"ub95_one_sided": ub, "geomean_ratio": ub * 0.97}


def test_decide_block():
    d = VA.decide_block(_p(0.7), 0, {"status": "pass"}, 0.82, "confirmatory")
    assert d["verdict"] == "positive_result_achieved" and d["confirmatory"] is True
    assert d["thresh_applicable"] is True
    for ub, fs in ((0.7, 0), (0.95, 0), (1.2, 3)):                     # descriptive: never a verdict
        d = VA.decide_block(_p(ub), fs, {"status": "pass"}, 0.82, "descriptive")
        assert d["confirmatory"] is False and d["block_status"] == "descriptive" and d["verdict"] == "descriptive"
        assert d["thresh_applicable"] is False and "positive" not in json.dumps(d)
        assert "failing_components" not in d and "sub_label" not in d and d["ub95"] == ub
    with pytest.raises(ValueError):                                     # same input validation as v11 decide
        VA.decide_block(_p(0.7), 0, {"status": None}, 0.82, "descriptive")
    with pytest.raises(ValueError):
        VA.decide_block(_p(0.7), 0, {"status": "pass"}, 0.82, "maybe")
    b = VA.decide_block(_p(0.95), 0, {"status": "pass"}, 0.9, "confirmatory")
    assert b["failing_components"] == ["rival_superiority_failed"] and b["sub_label"] == "faster_below_1"


def _row(m, s, e, n80, tau=1000, false=False, nstop=None, exh=0.0):
    return {"method": m, "seed": s, "eps": e, "N80_pen": n80, "tau_R": tau, "fwer_event": false,
            "false_by_k80": False, "n80_lt_tau": n80 < tau, "reached_stop": True, "validity": "rigorous",
            "exhaustion_at_k80": {"all": exh, "ctrl": exh, "treat": exh}, "schedule_digest": "s",
            "arrival_digest": f"a{s}", "N_stop_pen": nstop or min(tau, n80 * 1.2)}


def test_analyse_cell_v12_descriptives():
    seeds = list(range(30))
    rows = []
    for m, base in {VA.PRIMARY: 300, VA.RIVAL: 600, VA.HC_TUNED: 700, VA.RECT_HG: 550}.items():
        for s in seeds:
            if m == VA.RIVAL and s < 4:
                rows.append(_row(m, s, 5e-4, 1000, exh=1.0, nstop=1000))       # censored at tau
            else:
                rows.append(_row(m, s, 5e-4, base * (1 + 0.01 * (s % 5))))
    c = VA.analyse_cell_v12(rows, seeds, 5e-4, B=2000)
    u = c["descriptive_v12"]["uncensored_subset"]
    assert u["n_streams_in_subset"] == 26 and u["geomean_ratio"] == pytest.approx(0.5)
    assert u["subset_equals_rival_before_tau"] is True
    f = c["descriptive_v12"]["full_frontier"]
    assert f["rival_reaches_15_only_at_tau"] == 4 and f["n"] == 30
    assert c["primary_false_streams"] == 0


def test_registry_constants():
    from dsswm.stats import v11_analysis as V11
    assert VA.EVAL_SEEDS["women"] == tuple(range(39200, 39400)) and VA.EVAL_SEEDS["men"] == tuple(range(39400, 39600))
    assert not set(VA.EVAL_SEEDS["women"]) & set(VA.EVAL_SEEDS["men"])
    used = set(V11.EVAL_SEEDS) | set(range(30000, 39000))
    assert not used & (set(VA.EVAL_SEEDS["women"]) | set(VA.EVAL_SEEDS["men"]))
    assert VA.EPS_GRID == V11.EPS_GRID and VA.HC_GRID == V11.HC_GRID and VA.METHODS == V11.METHODS
    assert VA.K_GRID == 40 and VA.N80_K == 12 and VA.RULE_MIN_SUCCESS == 40 and VA.DEV_SEEDS == tuple(range(950, 1000))
    assert VA.THRESH_CAP == 0.90 and VA.THRESH_MARGIN == 0.10 and VA.SHARE_MAX == 0.5
    assert VA.select_eps is V11.select_eps and VA.thresh_rule is V11.thresh_rule and VA.decide is V11.decide
    from dsswm.stats.v6_analysis import B_BOOT, BOOT_SEED
    assert B_BOOT == 10_000 and BOOT_SEED == 42


# --------------------------------------------------------------------------------------------- runner
def test_runner_registry_and_code_files():
    import run_v12 as RV
    for c in OE.CAMPAIGNS:
        T = RV.TASKS[f"v12_{c}_full"]
        assert T["half"] == "eval" and T["seeds"] == list(VA.EVAL_SEEDS[c]) and T["campaign"] == c
        assert T["methods"] == list(VA.METHODS) and T["layer"] == OE.LAYERS[c]
    assert sorted(t for t, T in RV.TASKS.items() if T["half"] == "eval") == ["v12_men_full", "v12_women_full"]
    for t, T in RV.TASKS.items():
        if T["half"] == "dev":
            assert T["seeds"] == list(VA.DEV_SEEDS)
        if T["role"] == "eps_rule":
            assert not any(m.startswith(("TU-FDC", "FDC")) for m in T["methods"])
    assert RV.STOP_FRAC == 1.0 and RV.DELTA == 0.05
    code = Path(RV.CODE)
    for f in RV.CODE_FILES + RV.LOCK_ONLY_FILES:
        assert (code / f).exists(), f


def test_code_files_cover_every_imported_dsswm_module():
    import subprocess
    import run_v12 as RV
    code = Path(RV.CODE).resolve()
    prog = ("import sys, json, importlib; sys.path.insert(0, '.'); import run_v12 as RV\n"
            "for m in ('dsswm.envs.obd_v12_eval', 'dsswm.envs.obd_v11', 'dsswm.stats.prereg_v12', "
            "'dsswm.stats.v12_seal', 'dsswm.stats.v11_replica', 'dsswm.stats.v12_analysis', 'dsswm.baselines.fdc_dp', "
            "'dsswm.baselines.rect_dp', 'dsswm.baselines.rect_hg_dp', 'dsswm.envs.seg_v10', 'dsswm.stats.v6_replica', "
            "'dsswm.stats.v6_analysis'):\n"
            "    importlib.import_module(m)\n"
            "RV.make(RV.VA.RECT_HG, [1, 2]); RV.make(RV.VA.HC_NAMES[0], [1, 2])\n"
            "print(json.dumps(sorted(getattr(m, '__file__', '') or '' for n, m in sys.modules.items() "
            "if n.startswith('dsswm'))))")
    out = subprocess.run([sys.executable, "-c", prog], cwd=str(code), capture_output=True, text=True, check=True)
    loaded = {str(Path(f).resolve().relative_to(code)) for f in json.loads(out.stdout.strip().splitlines()[-1]) if f}
    loaded = {f for f in loaded if not f.endswith("__init__.py")}
    missing = sorted(loaded - set(RV.CODE_FILES))
    assert not missing, missing


def test_dev_stream_identity_and_cp_label():
    """One dev stream per method on the frozen women dev env: same schedule / arrival digest across methods, K = 40,
    RECT-HG-DP rows labelled checkpoint strength (never 'rigorous')."""
    import run_v12 as RV
    from dsswm.baselines.fdc_dp import DEFAULT_NODE_LIMIT, GRID_RATIO
    T = dict(RV.TASKS["v12_women_pilot"])
    cfg = {VA.PRIMARY: {"node_limit": DEFAULT_NODE_LIMIT, "grid_ratio": GRID_RATIO},
           VA.HC_TUNED: {"c": 0.5, "tf": 0.4}}
    RV.init_env(T, 1.5e-3, cfg)
    out = [RV.job((m, 950, 1.5e-3, "c", None, "d")) for m in VA.METHODS]
    assert all(r["error"] is None for r in out), [r["error"] for r in out]
    assert len({r["schedule_digest"] for r in out}) == 1 and len({r["arrival_digest"] for r in out}) == 1
    assert all(r["K_eval"] == 40 for r in out)
    hg = [r for r in out if r["method"] == VA.RECT_HG][0]
    assert hg["validity"] == "checkpoint_strength_CP" != "rigorous"
