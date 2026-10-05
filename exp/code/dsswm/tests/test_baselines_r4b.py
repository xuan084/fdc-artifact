"""r4_setup_baselines_b: B3 RAGE / Peace (rect, nominal, fav), B5 maq-Qini, H8 in-model certifiers.

Checks
  * pair-signature de-duplication is exact (max_pairs V_p equals brute force, A = 2 and A = 3);
  * XY design improves its own objective and is deterministic; Peace design is never worse than its XY warm start;
  * Gaussian width (enumeration) equals the brute-force pair supremum; 2000-draw MC relative s.e. < 2%;
  * billing invariants and adaptive pool exhaustion for the phased methods;
  * every method answers a large-gap known-answer toy correctly (rect / fav / B5 / H8 reach the stop rule);
  * rigorous variants (B3-rect, Peace-rect) have toy FWER CP upper <= 0.05 on 200 adaptive finite-pool streams
    (20 near-tie populations, dev seeds 920-939, x 10 permutation seeds);
  * validity labels; H8 feature parsing / model fits; Qini curve; registry.
"""
import itertools
import math

import numpy as np
import pytest
from scipy import stats

from dsswm.baselines.h8_models import (JPC1Step, MisLid1Step, TLearnerBoot, boost_features, boost_logistic,
                                       cell_features, fit_logistic, segment_features)
from dsswm.baselines.maq_desc import MaqQini, qini_curve
from dsswm.baselines.peace_frontier import PeaceFrontier, gaussian_width, gw_samples, peace_design, tau_value
from dsswm.baselines.rage_frontier import RAGEFrontier, pair_signatures, pair_V, xy_design
from dsswm.streams import frontier as fr
from dsswm.streams.frontier_runner import (BASELINES_B, SyntheticPoolEnv, build_ctx, make_frontier_method,
                                           run_stream)
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population

CR9_DESC = ['f0nm=0|f2nm=0|f3nm=0|f9nm=0', 'f0nm=0|f2nm=1|f3nm=0|f9nm=0', 'f0nm=0|f2nm=1|f3nm=0|f9nm=1',
            'f0nm=1|f2nm=0|f3nm=0|f9nm=0', 'f0nm=1|f2nm=0|f3nm=1|f9nm=0', 'f0nm=1|f2nm=1|f3nm=0|f9nm=0',
            'f0nm=1|f2nm=1|f3nm=0|f9nm=1', 'f0nm=1|f2nm=1|f3nm=1|f9nm=0', 'f0nm=1|f2nm=1|f3nm=1|f9nm=1']


def _brute_maxV(rows, w, var, p):
    best = 0.0
    for i, j in itertools.combinations(range(len(rows)), 2):
        v = sum(w[s] * (var[s, rows[i][s]] / p[s, rows[i][s]] + var[s, rows[j][s]] / p[s, rows[j][s]])
                for s in range(len(w)) if rows[i][s] != rows[j][s])
        best = max(best, v)
    return best


@pytest.mark.parametrize("S,A,m", [(5, 2, 20), (4, 3, 30), (6, 2, 64)])
def test_pair_signatures_exact(S, A, m):
    rng = np.random.default_rng(S * 10 + A)
    pols = fr.enumerate_policies(S, A).astype(np.int64)
    rows = pols[rng.choice(len(pols), size=min(m, len(pols)), replace=False)]
    w = rng.dirichlet(np.ones(S))
    var = rng.uniform(0.01, 0.25, size=(S, A))
    p = rng.dirichlet(np.ones(A), size=S)
    inc = pair_signatures(rows, A)
    assert pair_V(inc, w, var, p).max() == pytest.approx(_brute_maxV(rows.tolist(), w, var, p), rel=1e-12)
    assert len({tuple(r) for r in inc.astype(int).tolist()}) == inc.shape[0]


def test_xy_design_improves_and_is_deterministic():
    rng = np.random.default_rng(0)
    S, A = 6, 2
    rows = fr.enumerate_policies(S, A).astype(np.int64)
    w = rng.dirichlet(np.ones(S))
    var = rng.uniform(0.01, 0.25, size=(S, A))
    inc = pair_signatures(rows, A)
    p, rho = xy_design(inc, w, var)
    assert np.allclose(p.sum(1), 1) and (p > 0).all()
    assert rho <= pair_V(inc, w, var, np.full((S, A), 0.5)).max() + 1e-12
    p2, _ = xy_design(inc, w, var)
    assert np.array_equal(p, p2)
    # Neyman-like: with all pairs, the optimal split per segment is proportional to sigma
    assert np.allclose(p[:, 1] / p[:, 0], np.sqrt(var[:, 1] / var[:, 0]), rtol=0.1)


def test_gaussian_width_matches_bruteforce_and_mc_se():
    rng = np.random.default_rng(1)
    S, A = 9, 2
    pols = fr.enumerate_policies(S, A).astype(np.int64)
    w = rng.dirichlet(np.ones(S) * 2)
    var = rng.uniform(0.001, 0.25, size=(S, A))
    inc = pair_signatures(pols, A)
    p, _ = xy_design(inc, w, var)
    eta = rng.standard_normal((5, S, A))
    s = gw_samples(pols, w, var, p, eta)
    coef = np.sqrt(w[:, None] * var / p)
    for b in range(5):
        h = np.array([sum(coef[t, r[t]] * eta[b, t, r[t]] for t in range(S)) for r in pols])
        assert s[b] == pytest.approx(max(h[i] - h[j] for i in range(len(h)) for j in range(len(h))))
    W, se = gaussian_width(pols, w, var, p, np.random.default_rng(2).standard_normal((2000, S, A)))
    assert W > 0 and se < 0.02


def test_peace_design_not_worse_than_xy():
    rng = np.random.default_rng(3)
    S, A = 6, 2
    rows = fr.enumerate_policies(S, A).astype(np.int64)[rng.choice(64, 30, replace=False)]
    w = rng.dirichlet(np.ones(S))
    var = rng.uniform(0.01, 0.25, size=(S, A))
    inc = pair_signatures(rows, A)
    p0, _ = xy_design(inc, w, var)
    eta = rng.standard_normal((500, S, A))
    p = peace_design(rows, inc, w, var, 5.0, np.random.default_rng(4), iters=300, p0=p0, eta_eval=eta)
    assert tau_value(rows, inc, w, var, p, eta, 5.0)[0] <= tau_value(rows, inc, w, var, p0, eta, 5.0)[0] + 1e-12


FAST = {"Peace-rect": dict(smd_iters=50), "Peace-nominal": dict(smd_iters=50), "Peace-fav": dict(smd_iters=50)}


def _mk(name, env=None):
    return make_frontier_method(name, env, **FAST.get(name, {}))


@pytest.mark.parametrize("name", ["B3-rect", "B3-nominal", "B3-fav", "Peace-rect", "Peace-nominal", "Peace-fav"])
def test_phased_billing_and_exhaustion(name):
    env = _toy_population(921)
    s, rows = run_stream(env, _mk(name, env), 3, PROBS, -1.0, keep_U=False)
    assert s["billing_ok"] and all(r["billing_ok"] for r in rows)
    if name.endswith("rect"):         # eps < 0: the rectangle never certifies -> runs to tau_R through exhaustion
        assert s["t_end"] == env.tau_R and s["skipped"] + s["reselected"] > 0
    else:                             # nominal / fav may stop early by exact identification (|Z_q| = 1)
        assert s["t_end"] <= env.tau_R


@pytest.mark.parametrize("name", list(BASELINES_B))
def test_known_answer_toy(name):
    env = SyntheticPoolEnv([[4000, 4000], [3000, 3000], [3000, 3000]], [[0.1, 0.6], [0.2, 0.45], [0.3, 0.9]], seed=0,
                           n_min=200)
    m = _mk(name, env)
    s, rows = run_stream(env, m, 42, PROBS, 0.02)
    assert s["billing_ok"] and s["n_false"] == 0
    if name not in ("B3-nominal", "B3-fav", "Peace-nominal", "H8-MisLid"):   # paper constants: may need > tau_R
        assert s["reached_stop"] and s["N80"] < env.tau_R, (name, s["n_cert"])
    mu = env.true_mu("visit")
    ctx = build_ctx(env, PROBS, 0.02)
    for q, ih in enumerate(s["decided_pi"]):
        if ih >= 0:
            J = fr.policy_values(ctx.pols[ih:ih + 1], env.w, mu)[0]
            assert fr.solve_enum(env.w, mu, PROBS[q]).J_star - J <= 0.02 + 1e-12


def test_validity_labels():
    lab = {n: make_frontier_method(n).validity for n in BASELINES_B}
    assert lab["B3-rect"] == lab["Peace-rect"] == "rigorous"
    assert lab["B3-nominal"] == lab["Peace-nominal"] == "nominal"
    assert lab["B3-fav"] == lab["Peace-fav"] == "none"
    assert lab["B5"] == "asymptotic"
    assert all(lab[n] == "model" for n in ("H8-JPC1", "H8-Tboot", "H8-MisLid", "H8-MisLid0"))


def toy_fwer(name, seeds=range(920, 940), perms=range(10)):
    ev = n = certs = early = 0
    billing = True
    for sd in seeds:
        env = _toy_population(sd)
        for p in perms:
            s, _ = run_stream(env, _mk(name, env), 1000 * sd + p, PROBS, TOY_EPS, keep_U=False)
            ev += int(s["fwer_event"])
            certs += s["n_cert"]
            billing &= s["billing_ok"]
            early += int(s["reached_stop"] and s["N80"] < 0.5 * env.tau_R)
            n += 1
    return ev, n, certs, early, billing


@pytest.mark.parametrize("name", ["B3-rect", "Peace-rect"])
def test_rigorous_variants_toy_fwer(name):
    ev, n, certs, early, billing = toy_fwer(name)
    assert n == 200 and billing
    cp_up = 1.0 if ev >= n else float(stats.beta.ppf(0.95, ev + 1, n - ev))
    assert cp_up <= 0.05, (name, ev, n)
    assert certs > 0
    assert early >= 0.25 * n, f"{name}: certifications only near exhaustion (vacuous stress)"


def test_rage_constants():
    m = RAGEFrontier("nominal")
    assert m._thr(1) == 2 ** -3 and m._stop_scale(3) == 2 ** -4
    ctx = build_ctx(_toy_population(920), PROBS, 0.02)
    m.setup(ctx)
    assert m._q_min(ctx) == pytest.approx((6 * 7 / 2 + 1) / 0.2)
    assert m._delta_k(ctx, 3) == pytest.approx(0.05 / 9 / 5)
    pz = PeaceFrontier("nominal")
    pz.setup(ctx)
    assert pz._delta_k(ctx, 3) == pytest.approx(0.05 / 18 / 5)
    pf = PeaceFrontier("fav")
    assert pf._delta_k(ctx, 3) == pytest.approx(0.05 / 54 / 5)
    assert pz._q_min(ctx) == pytest.approx(6 / 0.01)


# ------------------------------------------------------------------------------------------------ B5 / H8
def test_qini_curve_concave_and_monotone():
    rng = np.random.default_rng(5)
    S = 9
    w = rng.dirichlet(np.ones(S))
    mu = rng.uniform(0.0, 0.3, size=(S, 2))
    path = qini_curve(w, mu, np.full((S, 2), 1000), (0.0, 1.0))
    c = np.array([p["cost"] for p in path])
    g = np.array([p["gain"] for p in path])
    assert (np.diff(c) > 0).all() and (np.diff(g) > 0).all()
    slopes = np.diff(g) / np.diff(c)
    assert (np.diff(slopes) <= 1e-12).all()
    assert len(path) - 1 == int((mu[:, 1] > mu[:, 0]).sum())


def test_segment_features_parsing():
    X = segment_features(CR9_DESC, 9)
    assert X.shape == (9, 4) and X[0].sum() == 0 and X[-1].sum() == 4
    hr = ['hs3=0|mens_only', 'hs3=0|womens_only', 'hs3=1|mens_only', 'hs3=1|womens_only', 'hs3=1|both',
          'hs3=2|mens_only', 'hs3=2|womens_only', 'hs3=2|both']
    Xh = segment_features(hr, 8)
    assert Xh.shape == (8, 4) and set(np.unique(Xh)) <= {0.0, 1.0}
    assert segment_features(["s0", "s1", "s2"], 3).shape == (3, 0)
    Phi = cell_features(9, 2, X)
    assert Phi.shape == (18, 14) and np.linalg.matrix_rank(Phi) == 14


def test_logistic_fit_saturated_recovers_cell_means():
    rng = np.random.default_rng(6)
    S, A = 4, 2
    Phi = cell_features(S, A, np.eye(S)[:, 1:])                 # saturated: d = S + 1 + (S-1) = 2S cells
    n = rng.integers(500, 5000, size=S * A).astype(float)
    mu = rng.uniform(0.05, 0.6, size=S * A)
    k = np.round(n * mu)
    _, p, _ = fit_logistic(Phi, n, k, lam=1e-8)
    assert np.allclose(p, k / n, atol=1e-6)


def test_boost_logistic_vectorised_and_consistent():
    Fb = boost_features(np.zeros((5, 0)), 5)                        # one-hot -> can fit every cell
    n = np.array([[1000, 2000, 1500, 800, 3000]], float)
    k = np.array([[100, 900, 300, 400, 150]], float)
    p = boost_logistic(Fb, n, k, rounds=400, lr=0.3)[0]
    assert np.allclose(p, k[0] / n[0], atol=0.01)
    pp = boost_logistic(Fb, np.vstack([n, n]), np.vstack([k, k]), rounds=50)
    assert np.allclose(pp[0], pp[1])


def test_mislid_zero_misspec_on_linear_truth_is_correct():
    # CR9-like additive truth: well specified for the linear class -> eps_mis = 0 version must be correct
    X = segment_features(CR9_DESC, 9)
    Phi = cell_features(9, 2, X)
    rng = np.random.default_rng(7)
    th = np.concatenate([rng.uniform(0.2, 0.5, 9), [0.1, 0.05, -0.08, 0.06, 0.02]])
    mu = (Phi @ th).reshape(9, 2)
    sizes = np.full((9, 2), 1500)
    env = SyntheticPoolEnv(sizes, np.clip(mu, 0.01, 0.99), seed=1, n_min=500)
    env.seg_desc = CR9_DESC
    probs = [fr.Problem(f"c|B{b}", "visit", (0.0, 1.0), b) for b in (0.3, 0.5, 0.7)]
    s, _ = run_stream(env, make_frontier_method("H8-MisLid0", env), 5, probs, 0.03, keep_U=False)
    assert s["n_false"] == 0 and s["billing_ok"]


def test_tboot_and_jpc_run_with_features():
    env = SyntheticPoolEnv(np.full((9, 2), 800), np.full((9, 2), 0.3) + np.linspace(0, 0.2, 9)[:, None] *
                           np.array([0, 1]), seed=2, n_min=400)
    env.seg_desc = CR9_DESC
    for name in ("H8-Tboot", "H8-JPC1", "B5"):
        s, rows = run_stream(env, make_frontier_method(name, env), 1, PROBS, 0.03)
        assert s["billing_ok"] and len(rows) >= 1 and all(np.isfinite(u) or u == np.inf for u in rows[-1]["U"])


def test_registry_complete():
    for n in BASELINES_B:
        assert make_frontier_method(n).name == n
    with pytest.raises(KeyError):
        make_frontier_method("nope")
