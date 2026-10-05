"""Round-3 mechanism layer: leverage / Lambda_perp / A_k / Rem bound / write-ahead log / AUC statistics.

Pre-registered unit tests (task r3_setup_mechanism):
  * Lin dynamic class: Lambda_perp = 0 exactly (<= 1e-10);   Lin-Static, same-exposure pairs: Lambda_perp > 0
  * noise-free identity dJ* - dJ_theta = <h, r> + Rem  (<= 1e-9; Rem by Gauss-Legendre, dJ by the original propagator)
  * predictable (only data before t), AST: no truth import in learner modules, GPU/CPU agreement (<= 1e-9)
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dsswm.certify.lin_closed import certify_lin
from dsswm.evidence.ellipsoid import EllipsoidSet
from dsswm.evidence.lr_set import SeqLRSet
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch
from dsswm.mechanism import oracle_check as oc
from dsswm.mechanism.leverage import (LinLeverage, NLLeverage, grid_halfwidths, orth_leverage, select_pairs,
                                      stiff_alignment, theta_hat_index)
from dsswm.mechanism.predictor_log import PredictorLog, ResultLog, done_keys, join, load_jsonl
from dsswm.models.lin_class import LinClass
from dsswm.models.nl_class import NLClass
from dsswm.stats import auc as A
from dsswm.streams.generator import (DEFAULT_NL_S_GRID, LIN_DEFAULTS, NL_DEFAULTS, make_lin_instance,
                                     make_nl_instance, nl_class_max_jtable)
from dsswm.streams.utilities import Utility

PKG = Path(__file__).resolve().parents[1]
LEARNER_NEW = ["mechanism/leverage.py", "mechanism/predictor_log.py", "mechanism/__init__.py", "stats/auc.py"]
FORBIDDEN = ("envs", "streams.generator", "generator", "streams.offgrid", "offgrid", "streams.lin_oos", "lin_oos",
             "oracle_check", "mechanism.oracle_check")
TRUTH_ATTRS = ("true_params", "true_theta_vector", "simulate_batch", "truth", "_alpha", "_beta", "_gamma", "_tau",
               "_lam", "_psi_left", "_syn", "env_truth")


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def nl():
    torch.set_num_threads(2)
    inst = make_nl_instance(734)
    prop = NLPropagator(2, 2, NL_DEFAULTS["nmax"], inst.env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"],
                        device="cpu")
    ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
    TH = NLLeverage(prop).theta_matrix(ncl.np_params)
    lev = NLLeverage(prop, theta_grid=TH)
    h = inst.env.handle()
    rng = np.random.default_rng(5)
    obs = list(inst.init_obs) + [h.step(int(rng.integers(inst.env.aspace.n))) for _ in range(80)]
    LT = prop.tables(ncl.torch_params())[0]
    lr = SeqLRSet(prop, LT, 0.05)
    for o in obs:
        lr.update(o)
    q = inst.problems[0]
    Jc = nl_class_max_jtable(q, prop, ncl.torch_params())
    mask = lr.mask().numpy()
    k_hat = theta_hat_index(lr.cum.numpy(), mask)
    return dict(inst=inst, prop=prop, ncl=ncl, TH=TH, lev=lev, obs=obs, lr=lr, q=q, J=Jc, mask=mask, k_hat=k_hat)


@pytest.fixture(scope="module")
def lin():
    inst = make_lin_instance(740)
    aspace = inst.env.aspace
    dyn = LinClass(3, 3, aspace.nb, nmax=LIN_DEFAULTS["nmax"])
    sta = LinClass(3, 3, aspace.nb, nmax=LIN_DEFAULTS["nmax"], static=True)
    h = inst.env.handle()
    rng = np.random.default_rng(3)
    obs = list(inst.init_obs) + [h.step(int(rng.integers(aspace.n))) for _ in range(400)]
    ell = EllipsoidSet(dyn, sigma=LIN_DEFAULTS["sigma"], delta=0.05, S=10.0)
    for o in obs:
        ell.update(o)
    return dict(inst=inst, aspace=aspace, dyn=dyn, sta=sta, obs=obs, ell=ell)


# ------------------------------------------------------------------------------------------------ NL: plumbing
def test_nl_mu_and_tables_match_propagator(nl):
    lev, prop, ncl, TH = nl["lev"], nl["prop"], nl["ncl"], nl["TH"]
    for k in (0, 777, 5000, ncl.B - 1):
        LT, EY = prop.tables(params_to_torch(ncl.params_at(k), device="cpu"))
        LT2, EY2 = lev.lt_from_mu(torch.as_tensor(lev.mu_from_theta(TH[k])))
        assert float((LT - LT2).abs().max()) < 1e-12 and float((EY - EY2).abs().max()) < 1e-12


def test_nl_J_mu_matches_jtable(nl):
    lev, q, ncl, TH = nl["lev"], nl["q"], nl["ncl"], nl["TH"]
    k = nl["k_hat"]
    Jm = lev.J_mu(lev.mu_from_theta(TH[k]), q).reshape(-1)
    assert np.max(np.abs(Jm - nl["J"][k])) < 1e-12


def test_nl_leverage_matches_finite_differences(nl):
    lev, q, TH = nl["lev"], nl["q"], nl["TH"]
    mu = lev.mu_from_theta(TH[nl["k_hat"]])
    g = lev.policy_leverage(mu, q, 0)
    used = np.flatnonzero(np.abs(g["grad"]) > 1e-8)
    assert len(used) > 3
    for x in used[:8]:
        e = 1e-6
        mp, mm = mu.copy(), mu.copy()
        mp[x] += e
        mm[x] -= e
        fd = float((lev.J_mu(mp, q, [0]) - lev.J_mu(mm, q, [0])).reshape(-1)[0]) / (2 * e)
        assert abs(fd - g["grad"][x]) < 1e-7
    # leverage lives only where the cell is drawn: h != 0 => d > 0
    assert np.all(g["occ"][used] > 0)


def test_nl_cell_counts_cover_ledger(nl):
    lev, obs, prop = nl["lev"], nl["obs"], nl["prop"]
    cnt = lev.cell_counts(obs)
    assert cnt.sum() == sum(len(prop.obs_factors(o)[0]) for o in obs)


# ------------------------------------------------------------------------------------------------ identity (T0(i))
@pytest.mark.parametrize("which", ["grid_truth", "offgrid_syn_truth"])
def test_nl_identity_noise_free(nl, which):
    lev, q, inst, ncl = nl["lev"], nl["q"], nl["inst"], nl["ncl"]
    truth = inst.env.true_params()
    if which == "offgrid_syn_truth":
        rng = np.random.default_rng(1)
        truth = {**truth, "alpha": truth["alpha"] + rng.uniform(-0.4, 0.4, 2), "syn": rng.uniform(-0.3, 0.3, (2, 2)),
                 "lam": truth["lam"] + 0.1}
    pairs = select_pairs(nl["J"], nl["mask"], nl["k_hat"], int(np.argmax(nl["J"][nl["k_hat"]])))
    cnt = lev.cell_counts(nl["obs"])
    for _, k1, k2 in pairs:
        chk = oc.nl_identity_check(lev, q, k1, k2, ncl.params_at(nl["k_hat"]), truth, xi=cnt / cnt.sum())
        assert chk["identity_err"] <= 1e-9, chk
        assert chk["plugin_consistency_err"] <= 1e-12


def test_nl_rem_bound_is_valid(nl):
    lev, q, TH = nl["lev"], nl["q"], nl["TH"]
    th = TH[nl["k_hat"]]
    for scale in (1.0, 0.2, 0.05):
        mu, lo, hi, a = lev.mu_box(th, lev.halfwidth * scale)
        assert np.all(lo <= mu + 1e-15) and np.all(mu <= hi + 1e-15)
        for k in (0, 1):
            rb = lev.rem_bound(mu, a, q, k)
            assert abs(rb["J"] - lev.J_mu(mu, q, [k]).reshape(-1)[0]) < 1e-12
            v = oc.rem_bound_validity(lev, q, k, mu, a, rb["rem_bar"], n_samples=40, seed=k)
            assert v["ratio"] <= 1.0 + 1e-9, (scale, k, v)
    # the bound is second order: shrinking the box by 10x shrinks it by >= ~50x
    _, _, _, a = lev.mu_box(th, lev.halfwidth * 0.1)
    _, _, _, a2 = lev.mu_box(th, lev.halfwidth * 0.01)
    r1, r2 = lev.rem_bound(lev.mu_from_theta(th), a, q, 0)["rem_bar"], lev.rem_bound(lev.mu_from_theta(th), a2, q, 0)["rem_bar"]
    assert r2 < r1 / 50


# ------------------------------------------------------------------------------------------------ projection algebra
def test_orth_leverage_pythagoras_and_need_data(nl):
    lev = nl["lev"]
    rng = np.random.default_rng(0)
    h = rng.standard_normal(lev.C)
    xi = rng.uniform(0.1, 1.0, lev.C)
    xi[3] = 0.0
    xi /= xi.sum()
    ol = orth_leverage(h, xi, lev.Phi)
    assert ol["need_data"] == [3]
    supp = xi > 0
    v = h[supp] / xi[supp]
    par = v - ol["resid"][supp]
    lhs = np.sum(xi[supp] * v ** 2)
    rhs = np.sum(xi[supp] * par ** 2) + ol["lambda_perp"] ** 2
    assert abs(lhs - rhs) < 1e-9 * lhs
    # h/xi in span phi  =>  Lambda_perp = 0
    beta = rng.standard_normal(lev.d)
    h2 = xi * (lev.Phi @ beta)
    assert orth_leverage(h2, xi, lev.Phi)["lambda_perp"] < 1e-10 * np.linalg.norm(beta)


def test_stiff_alignment_and_placebo_deterministic():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((6, 6))
    I = M @ M.T
    ev, V = np.linalg.eigh(I)
    u = V[:, -1] * 2.0
    s = stiff_alignment(u, I, k=3, blocks={"a": [0, 1, 2], "b": [3, 4, 5]})
    assert abs(s["A_k"] - 1.0) < 1e-12
    assert abs(s["w_k"] - 2.0 / np.sqrt(ev[-1])) < 1e-10
    assert s == stiff_alignment(u, I, k=3, blocks={"a": [0, 1, 2], "b": [3, 4, 5]})


# ------------------------------------------------------------------------------------------------ Lin layer
def _lin_problem(inst, k=0):
    return inst.problems[k]


def test_lin_dynamic_lambda_perp_zero(lin):
    llev = LinLeverage(lin["dyn"], ref=lin["dyn"], sigma=LIN_DEFAULTS["sigma"])
    worst = 0.0
    for q in lin["inst"].problems[:8]:
        Z = np.stack([lin["dyn"].z(p, q.loads0, q.H, q.utility, lin["aspace"]) for p in q.policies])
        cert = certify_lin(Z, lin["ell"], 0.05)
        vals = Z @ lin["ell"].theta_hat()
        vals[cert["pi_hat"]] = -np.inf
        pairs = [("a", cert["pi_hat"], int(np.argmax(vals))), ("b", cert["pi_hat"], cert["binding"])]
        rec = llev.predictors(q, lin["obs"], lin["ell"].theta_hat(), pairs, cert["r_bar"], 0.05, lin["aspace"])
        for p in rec["pairs"]:
            worst = max(worst, p["lambda_perp"])
            assert p["lambda_perp_nonparam"] >= 0
    assert worst <= 1e-10


def _same_exposure_problem(aspace, H=8, k=3):
    class OpenLoop:
        def __init__(self, steps):
            self.steps = set(steps)
            self.name = "T" + "".join("1" if t in self.steps else "0" for t in range(H))

        def act(self, t, loads, engaged):
            return aspace.index(((0, 0),)) if t in self.steps else aspace.index(())

    pols = [OpenLoop(range(k)), OpenLoop((0, 1, 5)), OpenLoop(range(0, 2 * k, 2))]
    util = Utility(w=np.ones(H), w_ret=0.0, c_q=1.0)
    return type("Q", (), {"policies": pols, "loads0": np.zeros(6, dtype=np.int64), "H": H, "utility": util,
                          "pid": "same_exposure"})()


def test_lin_static_same_exposure_lambda_perp_positive(lin):
    q = _same_exposure_problem(lin["aspace"])
    st = LinLeverage(lin["sta"], ref=lin["dyn"], sigma=LIN_DEFAULTS["sigma"])
    dy = LinLeverage(lin["dyn"], ref=lin["dyn"], sigma=LIN_DEFAULTS["sigma"])
    xi = st.cell_counts(lin["obs"])
    xi /= xi.sum()
    for k1, k2 in ((0, 1), (0, 2), (1, 2)):
        h = st.contrast_leverage(q, k1, k2, lin["aspace"])
        assert np.abs(st.Phi.T @ h).max() < 1e-12   # static value tie: d^s = 0
        assert np.abs(h).max() > 0                                                        # but h != 0 cellwise
        lam_s = orth_leverage(h, xi, st.Phi, st.Phi_ref)["lambda_perp"]
        lam_d = orth_leverage(h, xi, dy.Phi, dy.Phi_ref)["lambda_perp"]
        assert lam_s > 1e-3 and lam_d <= 1e-10, (lam_s, lam_d)


def test_lin_identity(lin):
    inst = lin["inst"]
    llev = LinLeverage(lin["dyn"], ref=lin["dyn"], sigma=LIN_DEFAULTS["sigma"])
    th_star = inst.env.true_theta_vector()
    for q in inst.problems[:5]:
        chk = oc.lin_identity_check(llev, q, 0, 1, lin["ell"].theta_hat(), th_star, lin["aspace"], lin["dyn"])
        assert chk["identity_err"] <= 1e-9


# ------------------------------------------------------------------------------------------------ predictability
def test_predictors_are_predictable(nl):
    lev, q, TH, obs = nl["lev"], nl["q"], nl["TH"], nl["obs"]
    t = 60
    pairs = [("a", 0, 1)]
    r1 = lev.predictors(q, obs[:t], TH[nl["k_hat"]], pairs, 0.05, with_rem=False)
    tampered = obs[:t] + list(reversed(obs[t:]))
    r2 = lev.predictors(q, tampered[:t], TH[nl["k_hat"]], pairs, 0.05, with_rem=False)
    assert r1["n_obs"] == t and r1["last_serial"] == obs[t - 1].serial
    for key in ("lambda_perp", "phi_perp", "A_k", "w_k", "rho_star"):
        assert r1["pairs"][0][key] == r2["pairs"][0][key]
    r3 = lev.predictors(q, obs[:t + 10], TH[nl["k_hat"]], pairs, 0.05, with_rem=False)
    assert r3["pairs"][0]["lambda_perp"] != r1["pairs"][0]["lambda_perp"]   # it does depend on the data it is given


# ------------------------------------------------------------------------------------------------ truth isolation
def _imports(path: Path):
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.append(("." * node.level) + mod)
            out += [("." * node.level) + mod + "." + a.name for a in node.names]
    return out


@pytest.mark.parametrize("rel", LEARNER_NEW)
def test_ast_no_truth_import_in_learner_modules(rel):
    path = PKG / rel
    for name in _imports(path):
        s = name.lstrip(".")
        assert not any(s == f or s.startswith(f + ".") or s.endswith("." + f) or s.endswith(f) for f in FORBIDDEN), \
            f"{rel} imports {name}"
        assert ".envs" not in name
    tree = ast.parse(path.read_text())
    hits = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in TRUTH_ATTRS
            and not (isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"))]
    assert not hits, f"{rel} accesses truth attributes {hits}"


def test_runtime_import_closure_excludes_truth_and_oracle():
    mods = ["dsswm.mechanism.leverage", "dsswm.mechanism.predictor_log", "dsswm.stats.auc"]
    code = ("import sys, importlib; sys.path.insert(0, %r)\n" % str(PKG.parent) +
            "for m in %r: importlib.import_module(m)\n" % mods +
            "bad=[m for m in sys.modules if m.startswith('dsswm.envs') or m in ('dsswm.streams.generator',"
            "'dsswm.streams.offgrid','dsswm.streams.lin_oos','dsswm.mechanism.oracle_check')]\n"
            "assert not bad, bad\nprint('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ------------------------------------------------------------------------------------------------ GPU / CPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_cpu_agreement(nl):
    inst, ncl, TH, q = nl["inst"], nl["ncl"], nl["TH"], nl["q"]
    pg = NLPropagator(2, 2, NL_DEFAULTS["nmax"], inst.env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"],
                      device="cuda")
    lg = NLLeverage(pg, theta_grid=TH)
    lc = nl["lev"]
    th = TH[nl["k_hat"]]
    pairs = [("a", 0, 1), ("b", 0, len(q.policies) - 1)]
    rc = lc.predictors(q, nl["obs"], th, pairs, 0.03)
    rg = lg.predictors(q, nl["obs"], th, pairs, 0.03)
    for a, b in zip(rc["pairs"], rg["pairs"]):
        for key in ("lambda_perp", "phi_perp", "A_k", "w_k", "rem_bar", "dJ_hat"):
            assert abs(a[key] - b[key]) <= 1e-9 * max(1.0, abs(a[key])), (key, a[key], b[key])
    hc = lc.contrast_leverage(lc.mu_from_theta(th), q, 0, 1)["h"]
    hg = lg.contrast_leverage(lg.mu_from_theta(th), q, 0, 1)["h"]
    assert np.max(np.abs(hc - hg)) <= 1e-9


# ------------------------------------------------------------------------------------------------ write-ahead log
def test_predictor_log_write_ahead(tmp_path):
    pl = PredictorLog(tmp_path / "predictors.jsonl")
    rl = ResultLog(tmp_path / "results.jsonl")
    key = {"instance": 734, "stream": 0, "method": "JPC", "arm": "full", "problem": "q0"}
    pl.write({**key, "lambda_perp": np.float64(0.3), "rho_star": float("inf"), "pairs": [{"A_k": 0.2}]})
    with pytest.raises(ValueError):
        pl.write({**key, "problem": "q1", "true_regret": 0.1})
    with pytest.raises(ValueError):
        pl.write({**key, "problem": "q2", "pairs": [{"eta_arg": 0.1}]})
    with pytest.raises(KeyError):
        pl.write({"instance": 1})
    rl.write({**key, "true_regret": 0.0, "false_cert": False})
    rows = join(tmp_path / "predictors.jsonl", tmp_path / "results.jsonl")
    assert len(rows) == 1 and rows[0]["pred"]["rho_star"] == "inf"
    assert done_keys(tmp_path / "results.jsonl") == {(734, 0, "JPC", "full", "q0")}
    # a result logged before its predictor violates write-ahead
    rl.write({**key, "problem": "q9"})
    pl.write({**key, "problem": "q9"})
    with pytest.raises(RuntimeError):
        join(tmp_path / "predictors.jsonl", tmp_path / "results.jsonl")
    # torn last line is tolerated (resumable)
    with open(tmp_path / "predictors.jsonl", "a") as fh:
        fh.write('{"instance": 7, "str')
    assert len(load_jsonl(tmp_path / "predictors.jsonl")) == 2
    assert json.loads((tmp_path / "predictors.jsonl").read_text().splitlines()[0])["phase"] == "pre_scoring"


# ------------------------------------------------------------------------------------------------ AUC statistics
def test_auc_basic():
    assert A.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert A.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert abs(A.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-12
    assert np.isnan(A.auc([1, 2], [0, 0]))
    assert abs(A.pr_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.8333333333333333) < 1e-12
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    s = np.round(rng.standard_normal(300), 1)
    y = rng.random(300) < 1 / (1 + np.exp(-2 * s))
    assert abs(A.auc(s, y) - roc_auc_score(y, s)) < 1e-12
    assert abs(A.pr_auc(s, y) - average_precision_score(y, s)) < 1e-12


def test_cluster_bootstrap_and_tests():
    rng = np.random.default_rng(1)
    n_cl, per = 40, 15
    cl = np.repeat(np.arange(n_cl), per)
    x = rng.standard_normal(n_cl * per)
    z = rng.standard_normal(n_cl * per)
    y = (rng.random(n_cl * per) < 1 / (1 + np.exp(-(1.5 * x - 0.5)))).astype(int)
    ci = A.auc_ci(x, y, cl, B=300)
    assert ci["lo"] < ci["est"] < ci["hi"] and ci["lo"] > 0.6
    d = A.delta_auc_ci(x, z, y, cl, B=300)
    assert d["lo"] > 0
    pr = A.pr_auc_ci(x, y, cl, B=200)
    assert pr["lo"] <= pr["est"] <= pr["hi"]
    lr = A.lr_test(z[:, None], np.column_stack([z, x]), y)
    assert lr["p"] < 1e-6 and lr["df"] == 1
    lr0 = A.lr_test(x[:, None], np.column_stack([x, z]), y)
    assert lr0["p"] > 1e-3


def test_clogit_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.discrete.conditional_models")
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(30), 8)
    X = rng.standard_normal((240, 2))
    off = rng.standard_normal(30)[g]
    y = (rng.random(240) < 1 / (1 + np.exp(-(X @ [1.0, -0.5] + off)))).astype(int)
    mine = A.clogit_fit(X, y, g)
    ref = sm.ConditionalLogit(y, X, groups=g).fit(disp=0)
    assert np.max(np.abs(mine["beta"] - ref.params)) < 1e-4
    assert abs(mine["loglik"] - ref.llf) < 1e-6
    t = A.clogit_lr_test(X[:, :1], X, y, g)
    assert t["df"] == 1 and t["p"] < 0.05
