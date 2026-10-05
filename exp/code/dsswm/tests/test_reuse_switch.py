"""Round-3 reuse switch, gap-quota streams, offset-null, MH-RR, lock v3 (r3_setup_reuse_switch)."""
import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dsswm.evidence.reuse_switch import (PROV_INITIAL, PROV_REAL, PROV_VOL, PROV_VOL_PAD, PROV_ORTH, REPLAY_PROVS,
                                         BillingError, ReuseSwitch, ShadowLedger, parse_arm, score_truncations)
from dsswm.stats.mh import mh_risk_ratio, stratified_mh_rr
from dsswm.stats.prereg import assert_locked, canonical_hash
from dsswm.streams.generator import make_nl_instance

PKG = Path(__file__).resolve().parents[1]
CACHE = PKG.parents[1] / "cache" / "jtables_r3"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 734


class DummySet:
    """Minimal confidence-set stand-in: counts rows."""

    def __init__(self):
        self.n_rounds = 0
        self.serials = []

    def update(self, obs):
        self.n_rounds += 1
        self.serials.append(obs.serial)


def _toy(n0=5):
    inst = make_nl_instance(3, K=0, n0=n0)
    sib = make_nl_instance(3, noise_seed=42 + 7919, K=0, n0=n0)
    return inst, sib


def _drive(sw, env, pids, steps_per):
    h = env.handle()
    rng = np.random.default_rng(0)
    accs = []
    for k, pid in enumerate(pids):
        lr = sw.begin_problem(k, pid)
        for _ in range(steps_per[k]):
            sw.record(h.step(int(rng.integers(env.aspace.n))))
        accs.append((lr, sw.end_problem(env.n_steps)))
    return accs


def _shadow(sib, pids, per=3):
    h = sib.env.handle()
    rows = [(h.step(0), pid) for pid in pids for _ in range(per)]
    return ShadowLedger(rows, pids, pad_fn=lambda n: [h.step(1) for _ in range(n)])


# ----------------------------------------------------------------------------------------- arms / billing
def test_parse_arm():
    assert parse_arm("off") == ("off", 0) and parse_arm("full") == ("full", 0)
    assert parse_arm("ev1") == ("ev", 1) and parse_arm("ev(20)") == ("ev", 20) and parse_arm("ev200") == ("ev", 200)
    with pytest.raises(ValueError):
        parse_arm("ev0")
    with pytest.raises(ValueError):
        parse_arm("probe")


@pytest.mark.parametrize("arm", ["off", "vol", "orth", "ev1", "ev20", "full"])
def test_billing_assertion_every_arm(arm):
    inst, sib = _toy()
    pids = ["p0", "p1", "p2"]
    shadow = _shadow(sib, pids) if arm == "vol" else None
    hs = sib.env.handle()
    orth = (lambda k, pid, n, ctx: [hs.step(2) for _ in range(n)]) if arm == "orth" else None
    sw = ReuseSwitch(arm, DummySet, inst.init_obs, shadow=shadow, orth_provider=orth)
    accs = _drive(sw, inst.env, pids, [4, 7, 2])
    assert inst.env.n_steps == sw.n_rounds_billed == 5 + 13
    for k, (lr, acc) in enumerate(accs):
        assert lr.n_rounds_total == lr.n_steps + lr.n_replay == lr.inner.n_rounds
        assert acc.new_env_steps == [4, 7, 2][k]
    n_old = [0, 4, 11]
    for k, (lr, acc) in enumerate(accs):
        if arm in ("vol", "orth"):
            assert acc.replay_steps == n_old[k]
        else:
            assert acc.replay_steps == 0


def test_unbilled_step_is_detected():
    inst, _ = _toy()
    sw = ReuseSwitch("full", DummySet, inst.init_obs)
    h = inst.env.handle()
    sw.begin_problem(0, "p0")
    sw.record(h.step(0))
    h.step(0)                                   # a platform round that bypassed the switch
    with pytest.raises(BillingError):
        sw.end_problem(inst.env.n_steps)


def test_replay_never_billed_and_auditable():
    inst, sib = _toy()
    pids = ["p0", "p1", "p2"]
    shadow = _shadow(sib, pids, per=2)          # sibling prefix shorter than N_k -> padding
    sib_before = sib.env.n_steps
    sw = ReuseSwitch("vol", DummySet, inst.init_obs, shadow=shadow)
    accs = _drive(sw, inst.env, pids, [5, 6, 1])
    billed = {o.serial for o, _ in sw.real_ledger}
    assert sum(sw.cost_by_problem.values()) == 12 and inst.env.n_steps == 5 + 12
    for lr, acc in accs:
        for r in lr.rows:
            assert (r.provenance in REPLAY_PROVS) == (r.serial not in billed)
    lr2, acc2 = accs[2]
    pc = acc2.prov_counts
    assert pc[PROV_VOL] == 4 and pc[PROV_VOL_PAD] == 11 - 4 and pc[PROV_INITIAL] == 5 and pc[PROV_REAL] == 1
    assert sib.env.n_steps == sib_before + 7    # padding came from the sibling platform, not the real one


def test_off_arm_discards_old_data():
    inst, _ = _toy()
    sw = ReuseSwitch("off", DummySet, inst.init_obs)
    accs = _drive(sw, inst.env, ["p0", "p1", "p2"], [3, 4, 5])
    init = {o.serial for o in inst.init_obs}
    for k, (lr, _) in enumerate(accs):
        own = {o.serial for o, pid in sw.real_ledger if pid == f"p{k}"}
        assert set(lr.inner.serials) == init | own
        assert lr.n_replay == 0
    inst2 = make_nl_instance(3, K=0, n0=5)
    full = ReuseSwitch("full", DummySet, inst2.init_obs)
    accs2 = _drive(full, inst2.env, ["p0", "p1", "p2"], [3, 4, 5])
    assert accs2[-1][0].inner.n_rounds == 5 + 12


def test_ev_m_gate():
    inst, _ = _toy()
    sw = ReuseSwitch("ev3", DummySet, inst.init_obs)
    h = inst.env.handle()
    sw.begin_problem(0, "p0")
    seen = []
    for _ in range(4):
        seen.append(sw.may_certify())
        sw.record(h.step(0))
    assert seen == [False, False, False, True]
    sw.end_problem(inst.env.n_steps)
    full = ReuseSwitch("full", DummySet, inst.init_obs)
    full.begin_problem(0, "q")
    assert full.may_certify()


def test_orth_infeasible_hook():
    inst, _ = _toy()
    sw = ReuseSwitch("orth", DummySet, inst.init_obs, orth_provider=lambda k, pid, n, ctx: None)
    assert sw.begin_problem(0, "p0") is None
    acc = sw.end_problem(inst.env.n_steps)
    assert acc.orth_feasible is False and acc.new_env_steps == 0


def test_shadow_ledger_exact_size():
    _, sib = _toy()
    pids = ["a", "b", "c"]
    sh = _shadow(sib, pids, per=3)
    assert len(sh.take(0, 0)) == 0
    t = sh.take(2, 5)
    assert len(t) == 5 and all(p == PROV_VOL for _, p, _ in t)
    t = sh.take(1, 7)
    assert [p for _, p, _ in t].count(PROV_VOL) == 3 and [p for _, p, _ in t].count(PROV_VOL_PAD) == 4
    t2 = sh.take(1, 7)
    assert [o.serial for o, _, _ in t] == [o.serial for o, _, _ in t2]    # deterministic prefix reuse


def test_truncation_scoring():
    s = score_truncations(2000, True)
    assert s["cost_tau1500"] == 1500 and s["censored_tau1500"] and s["cost_tau3000"] == 2000
    assert not s["censored_tau3000"]
    s = score_truncations(10, False)
    assert all(s[f"censored_tau{t}"] and s[f"cost_tau{t}"] == t for t in (1500, 3000, 6000))


# ----------------------------------------------------------------------------------------- gap quota / offset-null
@pytest.fixture(scope="module")
def builder():
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    ncl = NLClass(DEFAULT_NL_S_GRID, device=DEV)
    b = QuotaBuilder(SEED, ncl, None, CACHE)
    b.prop = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device=DEV)
    b.prop_cpu = NLPropagator(2, 2, 2, b.aspace, 1.0, 0.3, device="cpu")
    return b


def test_quota_layers_match_exact_gap(builder):
    from dsswm.exact.nl_propagate import NLPropagator, params_to_torch
    from dsswm.streams.gap_quota import gap_layer, top2_gap
    from dsswm.streams.utilities import Utility
    sel = builder.select("quota")
    assert not sel["quota_fail"] and sel["layer_counts"] == {"tie": 5, "near": 5, "clear": 5}
    st = builder.stream(0, "quota")
    prop = NLPropagator(2, 2, 2, builder.aspace, 1.0, 0.3, device="cpu")
    tp = params_to_torch(st.env.true_params(), device="cpu")
    for q, hm in zip(st.problems, st.harness):
        u = Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=q.utility.c_q)
        jt = prop.j_table(tp, q.policies, q.loads0, q.engaged0, q.H, u)[0]      # independent single-theta path
        d = top2_gap(jt)
        assert abs(d - hm["true_gap"]) <= 1e-9
        assert gap_layer(d) == hm["gap_layer"]


def test_offset_null_all_equal(builder):
    st = builder.stream(1, "offset_null")
    for q, J in zip(st.problems, st.J):
        jt = J[st.theta_index]
        assert float(jt.max() - jt.min()) <= 1e-12
        assert q.meta["offset_null"] and len(q.meta["offset_b"]) == len(q.policies)
        assert float((J.max(1) - J.min(1)).max()) > 0.02          # other class members still disagree
        assert J.min() >= -1e-12 and J.max() <= 1 + 1e-12


def test_stream_orders_and_crn(builder):
    a, b = builder.stream(0, "quota"), builder.stream(0, "quota")
    assert [q.pid for q in a.problems] == [q.pid for q in b.problems]
    assert [o.payload() for o in a.init_obs] != [] and a.init_obs[0].action == b.init_obs[0].action
    s1 = builder.stream(1, "quota")
    assert sorted(q.pid for q in s1.problems) == sorted(q.pid for q in a.problems)
    sib = builder.stream(0, "quota", sibling=True)
    assert sib.noise_seed == a.noise_seed + 7919 and [q.pid for q in sib.problems] == [q.pid for q in a.problems]


def test_no_tie_and_low_overlap(builder):
    st = builder.stream(0, "no_tie")
    assert len(st.problems) == 15 and all(h["true_gap"] >= 0.04 for h in st.harness)
    lo = builder.select("low_overlap")
    assert lo["layer_counts"] == {"tie": 5, "near": 5, "clear": 5}
    ov = builder.overlap(lo["base"])
    rnd = builder.overlap(builder.select("quota")["base"])
    mean = lambda o: np.mean([x["cos_max"] for x in o[1:]])  # noqa: E731
    assert mean(ov) <= mean(rnd) + 1e-12


def test_harness_cells_billing(builder):
    from dsswm.baselines.switched_nl import PublicNL
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.r3_harness import run_cell
    propc = NLPropagator(2, 2, 2, builder.aspace, 1.0, 0.3, device="cpu")
    pub = PublicNL(propc, NLClass(DEFAULT_NL_S_GRID, device="cpu"), [], [], 1.0, 2, 0.02, 0.05)
    hs = builder.stream(2, "quota", sibling=True).env.handle()
    orth = lambda k, pid, n, ctx: [hs.step(0) for _ in range(n)]  # noqa: E731  (mock design: billing only)
    for method, arms in (("JPC", ["off", "vol", "ev1", "full"]), ("B3", ["off", "full"]),
                         ("LR-chi2-grid", ["off", "full"]), ("B8", ["full"])):
        for arm in arms:
            rows = run_cell(builder, pub, 0, "quota", method, arm, tmax=400, n_problems=3)
            assert all(r["billing_ok"] for r in rows), (method, arm)
            assert rows[-1]["env_n_steps"] == 20 + sum(r["new_env_steps"] for r in rows)
            if arm == "vol":
                assert [r["replay_steps"] for r in rows] == [0] + list(np.cumsum([r["new_env_steps"] for r in rows])[:-1])
            if arm == "ev1":
                assert all(r["new_env_steps"] >= 1 for r in rows if r["status"] == "CERTIFIED")
    rows = run_cell(builder, pub, 0, "quota", "JPC", "orth", tmax=400, n_problems=2, orth_provider=orth)
    assert all(r["billing_ok"] and r["orth_feasible"] for r in rows)
    assert rows[1]["prov_counts"].get(PROV_ORTH, 0) == rows[0]["new_env_steps"]


# ----------------------------------------------------------------------------------------- truth isolation
LEARNER_NEW = ["evidence/reuse_switch.py", "baselines/switched_nl.py", "stats/mh.py", "stats/prereg.py"]
TRUTH_NEW = ("gap_quota", "r3_harness", "generator", "offgrid", "envs")


@pytest.mark.parametrize("rel", LEARNER_NEW)
def test_new_learner_modules_ast(rel):
    tree = ast.parse((PKG / rel).read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "")] + [(node.module or "") + "." + a.name for a in node.names]
        for n in names:
            assert not any(t in n.split(".") for t in TRUTH_NEW), f"{rel} imports {n}"
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("true_params", "theta_index", "harness", "true_theta_vector", "truth"), rel


def test_new_learner_modules_runtime():
    mods = ["dsswm.evidence.reuse_switch", "dsswm.baselines.switched_nl", "dsswm.stats.mh", "dsswm.stats.prereg"]
    code = ("import sys, importlib; sys.path.insert(0, %r)\n" % str(PKG.parent) +
            "for m in %r: importlib.import_module(m)\n" % mods +
            "bad=[m for m in sys.modules if m.startswith('dsswm.envs') or m.split('.')[-1] in %r]\n" % (TRUTH_NEW,) +
            "assert not bad, bad\nprint('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ----------------------------------------------------------------------------------------- stats
def test_mh_rr_known_value():
    rows = []
    # stratum A: exposed 2/10 events, unexposed 1/10; stratum B: exposed 4/10, unexposed 2/10  -> RR_MH = 2
    for s, (a, c) in {"A": (2, 1), "B": (4, 2)}.items():
        rows += [{"s": s, "e": 1, "y": int(i < a), "instance": i % 5} for i in range(10)]
        rows += [{"s": s, "e": 0, "y": int(i < c), "instance": i % 5} for i in range(10)]
    assert math.isclose(mh_risk_ratio(rows, "e", "y", ("s",)), 2.0)
    out = stratified_mh_rr(rows, "e", "y", ("s",), n_boot=200)
    assert math.isclose(out["point"], 2.0) and out["n_clusters"] == 5 and set(out["strata"]) == {"A", "B"}


# ----------------------------------------------------------------------------------------- lock v3
def _lock(tmp_path, **kw):
    lock = {"version": 3, "status": "locked", "x": 1, **kw}
    lock["sha256"] = canonical_hash(lock)
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(lock))
    return p


def test_assert_locked_v3(tmp_path):
    # round 4 bumped LOCK_VERSION to 4; the v3 path stays reproducible via an explicit version=3
    assert assert_locked(_lock(tmp_path), version=3)["version"] == 3
    with pytest.raises(RuntimeError, match="version 2"):
        assert_locked(_lock(tmp_path, version=2), version=3)
    with pytest.raises(RuntimeError, match="status"):
        assert_locked(_lock(tmp_path, status="provisional"), version=3)
    with pytest.raises(RuntimeError, match="status"):
        assert_locked(_lock(tmp_path, status="locked_no_eval"), version=3)
    p = _lock(tmp_path)
    d = json.loads(p.read_text()); d["x"] = 2; p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="hash"):
        assert_locked(p, version=3)
