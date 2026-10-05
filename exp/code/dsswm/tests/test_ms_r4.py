"""r4_setup_ms_baselines: multi-step harness (MS-H / MS-S / MS-R3 / MS-F) + Guard, MisLid-ms, DF-sep-ledger,
PolicyCert-tab, PERP-fact, always-dynamic JPC, FCC adapter."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

from dsswm.baselines.ms_common import CHECKPOINTS, TAU, T_MAX  # noqa: E402
from dsswm.streams import ms_r4 as M  # noqa: E402

BASE = Path(__file__).resolve().parents[1] / "baselines"
LEARNER_MODULES = ["ms_common.py", "guard.py", "mislid_ms.py", "dfsep_ledger.py", "policycert_tab.py", "perp_fact.py"]
FORBIDDEN_IMPORTS = ("ms_r4", "offgrid", "gap_quota", "replica", "run_r3", "run_r4", "generator", "envs")
FORBIDDEN_TOKENS = ("true_params", "J_true", "theta_index", "nl_r0_truth", "truth_dict")
R3_PILOT = M.WS / "exp" / "results" / "pilots" / "r3_hazard_a"
MSH_DIR = M.WS / "exp" / "results" / "pilots" / "r4_setup_ms_baselines" / "placements"


# ------------------------------------------------------------------------------------------------ static checks
def test_checkpoints_geometric():
    c = CHECKPOINTS
    assert c[0] == 0 and c[-1] == T_MAX
    inside = c[(c >= 1) & (c <= TAU)]
    assert len(inside) == 20 and inside[0] == 1 and inside[-1] == TAU
    assert np.all(np.diff(c) > 0)


@pytest.mark.parametrize("mod", LEARNER_MODULES)
def test_learner_never_imports_truth(mod):
    src = (BASE / mod).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            name = (node.module or "")
            assert not any(f in name for f in FORBIDDEN_IMPORTS), (mod, name)
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not any(f in a.name for f in FORBIDDEN_IMPORTS), (mod, a.name)
    code_only = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    body = ast.get_docstring(tree) or ""
    code_only = code_only.replace(body, "")
    for tok in FORBIDDEN_TOKENS:
        assert tok not in code_only, (mod, tok)


def test_handle_hides_truth():
    from dsswm.envs.base import TruthAccessError
    from dsswm.streams.gap_quota import make_env, nl_r0_truth
    h = make_env(750, nl_r0_truth(750), 42).handle()
    with pytest.raises(TruthAccessError):
        h.true_params()
    h.observable_state()


def test_horizon_never_shortened():
    from dsswm.streams.gap_quota import candidate_problem, make_env, nl_r0_truth
    aspace = make_env(750, nl_r0_truth(750), 42).aspace
    q = candidate_problem(750, 0, aspace)
    q.H = 3
    with pytest.raises(AssertionError):
        M.MSInstance("MS-H", 750, [q], [np.zeros(len(q.policies))], [{}], {"dyn_G1": [None]}, "dyn_G1", None)


# ------------------------------------------------------------------------------------------------ CS / solvers
def test_sparse_cs_matches_dense():
    from dsswm.baselines.policycert_tab import SparseBettingCS
    from dsswm.certify.fcc import BettingCS
    rng = np.random.default_rng(0)
    d, s = BettingCS(5, 0.01), SparseBettingCS(5, 0.01)
    for _ in range(400):
        c = int(rng.integers(0, 3))
        x = int(rng.random() < 0.3 + 0.2 * c)
        d.update(c, x)
        s.update(c, x)
    for a, b in zip(d.intervals(), s.intervals()):
        assert np.array_equal(a, b)
    assert s.intervals()[0][4] == 0.0 and s.intervals()[1][4] == 1.0


def _plan_setup():
    from dsswm.baselines.policycert_tab import TabularSolver
    from dsswm.certify.fcc import DecoupledSolver, cell_probs
    from dsswm.streams.gap_quota import candidate_problem, make_env, nl_r0_truth
    truth = nl_r0_truth(750)
    env = make_env(750, truth, 42)
    q = candidate_problem(750, 1, env.aspace)
    fs = DecoupledSolver(2, 2, 2, env.aspace, 0.3)
    ts = TabularSolver(2, 2, 2, env.aspace, 0.3)
    p = cell_probs({**env.true_params(), "psi_left": env.true_params()["psi_left"]}, fs.cells, 1.0)
    return q, fs, ts, p


def test_tabular_solver_matches_factor_at_zero_width_and_is_wider():
    q, fs, ts, p = _plan_setup()
    w, wr, cq = np.asarray(q.utility.w, float), float(q.utility.w_ret), 1.0
    for pol in q.policies[:3]:
        pf, pt = fs.plan(pol, q.loads0, q.engaged0, q.H), ts.plan(pol, q.loads0, q.engaged0, q.H)
        tab = np.full(ts.n_tab, 0.5)
        for cf, ct in zip(pf.cellids, pt.cellids):
            m = cf < fs.cells.n
            assert np.array_equal(m, ct < ts.n_tab)
            tab[ct[m]] = p[cf[m]]
        jf = fs.solve(pf, w, wr, cq, p, p)
        jt = ts.solve(pt, w, wr, cq, tab, tab)
        assert abs(jf[0] - jt[0]) < 1e-9 and abs(jf[1] - jt[1]) < 1e-9
        lo, hi = np.clip(p - 0.1, 0, 1), np.clip(p + 0.1, 0, 1)
        tlo, thi = np.full(ts.n_tab, 0.0), np.full(ts.n_tab, 1.0)
        for cf, ct in zip(pf.cellids, pt.cellids):
            m = cf < fs.cells.n
            tlo[ct[m]], thi[ct[m]] = lo[cf[m]], hi[cf[m]]
        bf, bt = fs.solve(pf, w, wr, cq, lo, hi), ts.solve(pt, w, wr, cq, tlo, thi)
        assert bt[0] <= bf[0] + 1e-12 and bt[1] >= bf[1] - 1e-12


def test_mislid_class_cell_probs_match_fcc():
    from dsswm.baselines.mislid_ms import class_cell_probs
    from dsswm.certify.fcc import CellIndex, cell_probs
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
    cells = CellIndex(2, 2, 2, ncl.nb)
    idx = [0, 77, 5000, 13823]
    sub = {k: v[idx] for k, v in ncl.np_params.items()}
    CP = class_cell_probs(sub, cells, 1.0)
    for r, b in enumerate(idx):
        pr = {k: v[b] for k, v in ncl.np_params.items()}
        assert np.allclose(CP[r], cell_probs(pr, cells, 1.0), atol=1e-12)


# ------------------------------------------------------------------------------------------------ harness runs
def _r3_inst(seed=750):
    plc = json.loads((R3_PILOT / "parts" / f"place_i{seed}.json").read_text())["R1adv"]
    return M.build_msr3(seed, plc)


@pytest.mark.skipif(not (R3_PILOT / "placements.json").exists(), reason="r3 pilot artefacts missing")
def test_r3_jpc_replay_identical_prefix():
    inst = _r3_inst(750)
    rows, info = M.run_stream(inst, "JPC", n_problems=3)
    old = [json.loads(l) for l in open(R3_PILOT / "results.jsonl")]
    old = sorted([r for r in old if r["instance"] == 750 and r["layer"] == "R1adv" and r["base_method"] == "JPC"
                  and r["stream"] == 0], key=lambda r: r["problem"])
    for a, b in zip(rows, old):
        assert (a["pid"], a["status"], a["new_env_steps"], a["certified_policy"], a["false_cert"]) == \
               (b["pid"], b["status"], b["new_env_steps"], b["certified_policy"], b["false_cert"])
    assert info["write_ahead_ok"]


@pytest.mark.skipif(not (R3_PILOT / "placements.json").exists(), reason="r3 pilot artefacts missing")
@pytest.mark.parametrize("method", [m for m in M.METHODS if m != "JPC"])
def test_every_method_billing_and_write_ahead(method, tmp_path):
    inst = _r3_inst(750)
    rows, info = M.run_stream(inst, method, tmax=150, n_problems=2, decision_path=tmp_path / "dec.jsonl")
    assert len(rows) == 2 and info["write_ahead_ok"]
    assert all(r["billing_ok"] for r in rows)
    assert all(r["env_n_steps"] == r["n_rounds_billed"] for r in rows)
    assert all(r["new_env_steps"] <= 150 for r in rows)
    lines = (tmp_path / "dec.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for r in rows:
        if r["status"] == "CERTIFIED":
            assert r["certified_policy"] is not None
    if method.startswith("Guard_c"):
        c = float(method[len("Guard_c"):])
        assert all(abs(r["x_eps_cert"] - M.EPS / c) < 1e-15 for r in rows)


def test_dfsep_fingerprint_and_ledger_reuse():
    from dsswm.baselines.dfsep_ledger import fingerprint
    from dsswm.certify.fcc import DecoupledSolver
    from dsswm.streams.gap_quota import candidate_problem, make_env, nl_r0_truth
    env = make_env(750, nl_r0_truth(750), 42)
    q = candidate_problem(750, 2, env.aspace)
    s = DecoupledSolver(2, 2, 2, env.aspace, 0.3)
    k1 = fingerprint(s, q.policies[0], q.loads0, q.engaged0, q.H)
    assert k1 == fingerprint(s, q.policies[0], q.loads0, q.engaged0, q.H)
    keys = {fingerprint(s, p, q.loads0, q.engaged0, q.H) for p in q.policies}
    assert len(keys) >= 2


@pytest.mark.skipif(not MSH_DIR.exists(), reason="MS-H placements not produced yet")
def test_msh_placements_truth_label_filter():
    files = sorted(MSH_DIR.glob("msh_i*.json"))
    assert files
    for f in files:
        d = json.loads(f.read_text())
        assert d["proposals"] <= M.MAX_PROPOSALS
        assert len(d["problems"]) == d["fill"] <= M.Q
        for r in d["per_problem"]:
            assert r["eta_arg_near_exante"] / M.EPS >= 1.0
            assert r["mu_flip"] >= 1.0 / M.MC
