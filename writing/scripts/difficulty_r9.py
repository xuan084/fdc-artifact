#!/usr/bin/env python3
"""r9 decision-difficulty summary (external review r9, P1-05).  Analysis of EXISTING data only; no new stream is run.

Two kinds of numbers, labelled everywhere:
  DEV  -- computed here on the DEVELOPMENT half of each lock with that lock's own frozen design loader
          (segmentation, costs, budgets, true cell means of the dev half).  No evaluation-half outcome is read.
  EVAL -- counted from SEALED evaluation rows already in exp/results/full (results.jsonl of the registered
          tasks); only per-stream fields the sealed runner recorded (N80_pen, tau_R, n80_lt_tau,
          exhaustion_at_k80) are read.  No statistic is computed on evaluation-half outcomes / evaluation truth.

Definitions (eps-optimal: J*_q - J(pi) <= eps on the DEV truth; feasible: cost(pi) <= B_q):
  (a) pooled share  = sum_q #{feasible eps-optimal pi} / sum_q #{feasible pi}            (S12 census definition)
  (b) mean share    = mean_q  #{feasible eps-optimal pi} / #{feasible pi}                 (v12 block-status gate,
                      dsswm.envs.obd_v11_eval.eps_opt_share / dsswm.stats.v12_analysis.block_status)
  (c) all-control   = #{q : J*_q - J(all-control) <= eps}                                 (v12 gate criterion ii)
Policy classes with 2^S members are counted EXACTLY: S <= 16 by enumeration, S = 32 by meet-in-the-middle over the
two 16-segment halves, S = 64 by an exact branch-and-bound count over the cardinality structure (all v10 costs are
8 units, so problem q is "treat at most floor(B_q / 8) segments"); subtrees whose every completion is (or no
completion can be) eps-optimal are counted in closed form with binomial sums.  If the node limit were hit the cell
is hit, a rigorous bracket from an exact integer DP on rounded-down / rounded-up uplifts is reported instead
(labelled as a bracket; 'not enumerable' if the bracket were wider than the printed precision).

Usage (cwd anywhere):  .venv/bin/python3 iter_001/writing/scripts/difficulty_r9.py [--tex-only | --check-eval]
  --tex-only    re-render the two LaTeX fragments from the stored json (no recomputation)
  --check-eval  recompute only the EVAL counts from the sealed row files present (results.jsonl or results.jsonl.gz)
                and compare them with the stored json; exit 1 on any difference or missing source; writes nothing
Outputs: writing/r9_difficulty.json, writing/r9_difficulty.md, writing/supplement/r9_tables/difficulty.tex,
         writing/latex_acm/r9_difficulty_main.tex
"""
from __future__ import annotations

import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

IT = Path(__file__).resolve().parents[2]          # iter_001
CODE = IT / "exp/code"
RES = IT / "exp/results"
FULL = RES / "full"
PLAN = IT / "plan"
sys.path.insert(0, str(CODE))
sys.setrecursionlimit(10000)

TOL = 1e-12               # gap <= eps + TOL (verify_r4 census); boundary pairs are counted and reported separately
BNB_NODE_LIMIT = 20_000_000

# ------------------------------------------------------------------------------------------------- block registry
BLOCKS = [
    dict(id="CR9", label="Criteo, S=9", short="Criteo, 9", locks="v7 / v9", kind="enum", layer="CR9", eps=0.001,
         role="confirmatory"),
    dict(id="X9", label="X5, S=9", short="X5, 9", locks="v8 / v9", kind="enum", layer="X9", eps=0.02,
         role="confirmatory"),
    dict(id="X5-16", label="X5, S=16", short="X5, 16", locks="v10", kind="v10", data="x5", S=16, eps=0.03,
         role="confirmatory", task="v10a_full_s16", analysis="v10_analysis_A.json"),
    dict(id="X5-32", label="X5, S=32", short="X5, 32", locks="v10", kind="v10", data="x5", S=32, eps=0.04,
         role="confirmatory", task="v10a_full_s32", analysis="v10_analysis_A.json"),
    dict(id="X5-64", label="X5, S=64 (descriptive)", short="X5, 64$^\\dagger$", locks="v10", kind="v10", data="x5",
         S=64, eps=0.05, role="descriptive", task="v10d_full_x5s64", analysis="v10_analysis_D.json"),
    dict(id="LE-16", label="Lenta, S=16", short="Lenta, 16", locks="v10", kind="v10", data="lenta", S=16, eps=0.004,
         role="confirmatory", task="v10b_full_s16", analysis="v10_analysis_B.json"),
    dict(id="LE-32", label="Lenta, S=32", short="Lenta, 32", locks="v10", kind="v10", data="lenta", S=32, eps=0.006,
         role="confirmatory", task="v10b_full_s32", analysis="v10_analysis_B.json"),
    dict(id="LE-64", label="Lenta, S=64", short="Lenta, 64", locks="v10", kind="v10", data="lenta", S=64, eps=0.008,
         role="confirmatory", task="v10b_full_s64", analysis="v10_analysis_B.json"),
    dict(id="OBD", label="Open Bandit random/all, S=9", short="OBD all", locks="v11", kind="v11", eps=3e-4,
         role="confirmatory", analysis="v11_obd/v11_analysis.json"),
    dict(id="OBD-W", label="Open Bandit random/women, S=8 (descriptive)", short="OBD women$^\\dagger$", locks="v12",
         kind="v12", campaign="women", eps=7.5e-4, role="descriptive", analysis="v12_obd/v12_women_analysis.json"),
    dict(id="OBD-M", label="Open Bandit random/men, S=10 (descriptive)", short="OBD men$^\\dagger$", locks="v12",
         kind="v12", campaign="men", eps=1e-3, role="descriptive", analysis="v12_obd/v12_men_analysis.json"),
]

# Sealed evaluation sources: (file, primary, governing rival, rival-role note, analysis file that names the rival)
EVAL_SRC = {
    "CR9": [dict(lock="v7", file="v7a_full_b/results.jsonl", eps=0.001, primary="FDC-BF", rival="HC-WoR",
                 why="governing_rival in v7_analysis_A.json (UB* 0.670 < 0.80)", analysis="v7_analysis_A.json"),
            dict(lock="v9", file="v9a_full_d/results.jsonl", eps=0.001, primary="TU-FDC", rival="HC-WoR",
                 why="governing_rival HC-WoR@D in v9_analysis_A.json (UB* 0.723 < 0.85)",
                 analysis="v9_analysis_A.json")],
    "X9": [dict(lock="v8", file="v8a_full_a/results.jsonl", eps=0.02, primary="FDC-BF", rival="RECT-ck-HG",
                rival_ni="PJC-local",
                why="governing_rival_F RECT-ck-HG (UB* 0.368 < 0.60); NI rival governing_rival_G PJC-local "
                    "(1.011 < 1.05) in v8_analysis_A.json", analysis="v8_analysis_A.json"),
           dict(lock="v9", file="v9b_full_d/results.jsonl", eps=0.02, primary="TU-FDC", rival="HC-WoR",
                rival_ni="TU-PJC",
                why="governing_rival HC-WoR@D (0.379 < 0.50); governing_NI_rival TU-PJC@D (1.009 < 1.05) in "
                    "v9_analysis_B.json", analysis="v9_analysis_B.json")],
}
for _b in BLOCKS:
    if _b["kind"] == "v10":
        EVAL_SRC[_b["id"]] = [dict(lock="v10", file=f"{_b['task']}/results.jsonl", eps=_b["eps"],
                                   primary="TU-FDC-DP(b)", rival="RECT-BF-DP-TU",
                                   why=f"registered rival of every lock-v10 cell ({_b['analysis']})",
                                   analysis=_b["analysis"])]
EVAL_SRC["OBD"] = [dict(lock="v11", file="v11_obd/v11_obd_full/results.jsonl", eps=3e-4, primary="TU-FDC-DP(b)",
                        rival="RECT-BF-DP-TU", why="registered rival (v11_obd/v11_analysis.json decision rule)",
                        analysis="v11_obd/v11_analysis.json")]
EVAL_SRC["OBD-W"] = [dict(lock="v12", file="v12_obd/v12_women_full/results.jsonl", eps=7.5e-4,
                          primary="TU-FDC-DP(b)", rival="RECT-BF-DP-TU",
                          why="primary rival of the descriptive block (v12_obd/v12_women_analysis.json)",
                          analysis="v12_obd/v12_women_analysis.json")]
EVAL_SRC["OBD-M"] = [dict(lock="v12", file="v12_obd/v12_men_full/results.jsonl", eps=1e-3, primary="TU-FDC-DP(b)",
                          rival="RECT-BF-DP-TU",
                          why="primary rival of the descriptive block (v12_obd/v12_men_analysis.json)",
                          analysis="v12_obd/v12_men_analysis.json")]

# Published numbers that must be reproduced before any new number is trusted
PUBLISHED = {
    "CR9": {"pooled_n_feasible": 3386, "pooled_n_eps_opt": 437,
            "src": "supplement S12 / verify_r4_numbers.py census"},
    "X9": {"pooled_n_feasible": 3347, "pooled_n_eps_opt": 3076, "src": "supplement S12 / verify_r4_numbers.py census"},
    "OBD": {"mean_share": 0.14164758798363972, "n_allctrl": 0,
            "src": "exp/results/pilots/v11_dev/v11_analysis_dev.json eps_optimal_share (dev)"},
    "OBD-W": {"mean_share": 0.6611021276803647, "n_allctrl": 3,
              "src": "exp/results/v12_gates/v12_women_block_status.json (S23: 0.661, 3 of 15)"},
    "OBD-M": {"mean_share": 0.8575499647604637, "n_allctrl": 5,
              "src": "exp/results/v12_gates/v12_men_block_status.json (S23: 0.858, 5 of 15)"},
}


def J(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------------------------------------- dev loaders
def dev_enum(layer, eps):
    """Locks v7/v8/v9 (S = 9, 512 enumerated policies): run_r5s_v8.init_env, the loader the S12 census used."""
    import run_r5s_v8 as R8
    R8.init_env(layer, "dev", (eps,))
    E = R8._ENV
    ctx, Jv, Js, env = E["ctxs"][eps], np.asarray(E["J"]), np.asarray(E["Js"]), E["env"]
    pols = np.asarray(ctx.pols)
    ac = np.flatnonzero((pols == 0).all(1))
    assert len(ac) == 1
    ac = int(ac[0])
    per_q, Mtot, Gtot, nbd, nac = [], 0, 0, 0, 0
    for q in range(ctx.Q):
        f = np.asarray(ctx.feas[q], bool)
        gap = Js[q] - Jv[f]
        g = int((gap <= eps + TOL).sum())
        nbd += int((np.abs(gap - eps) <= 1e-9).sum())
        Mtot += int(f.sum())
        Gtot += g
        per_q.append(dict(n_feasible=int(f.sum()), n_eps_opt=g, share=g / int(f.sum())))
        assert f[ac]
        nac += int(Js[q] - Jv[ac] <= eps + TOL)
    J0 = float(Jv[ac])
    base = float(np.asarray(env.outcomes_view(E["outcome"])).mean())
    return dict(S=int(env.S), A=int(env.A), n_policies=int(len(pols)), per_problem=per_q, pooled_n_feasible=Mtot,
                pooled_n_eps_opt=Gtot, n_allctrl=nac, J0=J0, Jstar=[float(x) for x in Js], dev_base_rate=base,
                boundary_within_1e_9=nbd, method="enumeration (512 policies, ctx.feas)",
                loader=f"run_r5s_v8.init_env('{layer}', 'dev', ({eps},))")


def dev_env_seg(b):
    """Lock v10 / v11 / v12 dev envs through each lock's own frozen-design class, sha-checked against the lock."""
    from dsswm.envs.seg_v10 import true_opt
    if b["kind"] == "v10":
        from dsswm.envs.seg_v10_eval import SegEnvV10Frozen
        lock = J(PLAN / "prereg_lock_v10_addendum.json")
        env = SegEnvV10Frozen(b["data"], b["S"], "dev", frozen_sha256=lock["frozen_segmentation"]["seg_v10_frozen.json"])
        loader = f"SegEnvV10Frozen('{b['data']}', {b['S']}, 'dev', frozen_sha256=<lock v10>)"
    elif b["kind"] == "v11":
        from dsswm.envs.obd_v11_eval import OBDEnvV11Frozen
        lock = J(PLAN / "prereg_lock_v11_addendum.json")
        env = OBDEnvV11Frozen("dev", frozen_sha256=lock["frozen_gates"]["exp/results/v11_gates/obd_v11_frozen.json"])
        loader = "OBDEnvV11Frozen('dev', frozen_sha256=<lock v11>)"
    else:
        from dsswm.envs.obd_v12_eval import OBDEnvV12Frozen
        lock = J(PLAN / "prereg_lock_v12_addendum.json")
        key = f"exp/results/v12_gates/obd_v12_{b['campaign']}_frozen.json"
        env = OBDEnvV12Frozen(b["campaign"], "dev", frozen_sha256=lock["frozen_gates"][key])
        loader = f"OBDEnvV12Frozen('{b['campaign']}', 'dev', frozen_sha256=<lock v12>)"
    assert env.half == "dev"
    Js, mu = true_opt(env)
    return env, np.asarray(Js), np.asarray(mu), loader


# ------------------------------------------------------------------------------------------------- exact counters
def count_enum(w, mu, cost, B, T):
    """Enumerate all A^S = 2^S policies (S <= 16), as obd_v11_eval.eps_opt_share does."""
    S = len(w)
    pols = ((np.arange(2 ** S)[:, None] >> np.arange(S)[None, :]) & 1).astype(np.int64)
    val = (w[None, :] * mu[np.arange(S)[None, :], pols]).sum(1)
    cst = cost[np.arange(S)[None, :], pols].sum(1)
    f = cst <= B
    vf = val[f]
    return int(f.sum()), int((vf >= T - TOL).sum()), int((np.abs(vf - T) <= 1e-9).sum())


def _half_table(w, mu, cost, idx):
    m = len(idx)
    pols = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1).astype(np.int64)
    val = (w[idx][None, :] * mu[idx][np.arange(m)[None, :], pols]).sum(1)
    cst = cost[idx][np.arange(m)[None, :], pols].sum(1)
    return val, cst


def count_mitm(w, mu, cost, B, T, tabs=None):
    """Exact count over 2^S policies for S = 32: split into two 16-segment halves; for each cost level of half A,
    count half-B members with cost <= B - cA and value >= T - vA (sorted arrays + searchsorted)."""
    S = len(w)
    if tabs is None:
        h = S // 2
        tabs = (_half_table(w, mu, cost, np.arange(h)), _half_table(w, mu, cost, np.arange(h, S)))
    (va, ca), (vb, cb) = tabs
    n_feas = n_good = n_bd = 0
    for c in np.unique(ca):
        rem = B - c
        if rem < 0:
            continue
        sel = np.sort(vb[cb <= rem])
        if len(sel) == 0:
            continue
        a = va[ca == c]
        n_feas += len(a) * len(sel)
        n_good += int((len(sel) - np.searchsorted(sel, T - TOL - a, side="left")).sum())
        n_bd += int((np.searchsorted(sel, T + 1e-9 - a, side="right")
                     - np.searchsorted(sel, T - 1e-9 - a, side="left")).sum())
    return int(n_feas), int(n_good), int(n_bd), tabs


def count_card_bnb(J0, u, k, T, node_limit=BNB_NODE_LIMIT):
    """Exact count of subsets X of {0..S-1} with |X| <= k and J0 + sum_{s in X} u_s >= T - TOL.
    Items are processed in decreasing |u|; a subtree is closed when even its best completion fails (count 0) or its
    worst completion succeeds (count = sum_{j <= r} C(n_rem, j)).  Returns (count, nodes) or (None, nodes)."""
    S = len(u)
    order = np.argsort(-np.abs(u), kind="stable")
    uu = [float(x) for x in u[order]]
    # suffix prefix sums of positives (desc) and negatives (most negative first); |u|-order keeps both sorted
    pos_cum, neg_cum = [], []
    for i in range(S + 1):
        p = [x for x in uu[i:] if x > 0]
        n = [x for x in uu[i:] if x < 0]
        pos_cum.append(np.concatenate([[0.0], np.cumsum(p)]).tolist())
        neg_cum.append(np.concatenate([[0.0], np.cumsum(n)]).tolist())
    cle = [[sum(math.comb(n, j) for j in range(r + 1)) for r in range(S + 1)] for n in range(S + 1)]
    thr = T - TOL
    nodes = [0]

    def rec(i, r, s):
        nodes[0] += 1
        if nodes[0] > node_limit:
            raise OverflowError
        pc, nc = pos_cum[i], neg_cum[i]
        if s + pc[min(r, len(pc) - 1)] < thr:
            return 0
        if s + nc[min(r, len(nc) - 1)] >= thr:
            return cle[S - i][r]
        c = rec(i + 1, r, s)
        if r > 0:
            c += rec(i + 1, r - 1, s + uu[i])
        return c

    try:
        return rec(0, k, J0), nodes[0]
    except OverflowError:
        return None, nodes[0]


def card_dp_bracket(J0, u, ks, Ts, h):
    """Rigorous bracket for the S = 64 cardinality classes when branch-and-bound hits its node limit.
    Each u_s is rounded DOWN (resp. UP) to the grid h*Z; an exact integer DP over (number treated, rounded sum) then
    counts, for every problem q, the subsets with |X| <= k_q whose rounded sum clears the threshold.  Rounding down
    can only lose eps-optimal subsets and rounding up can only add some, so lo <= true count <= hi (exact uint64
    counts; every count is <= sum_{j<=51} C(64, j) < 2^64)."""
    S, kmax = len(u), int(max(ks))
    out = {}
    for mode, fn in (("lo", np.floor), ("hi", np.ceil)):
        iu = fn(np.asarray(u) / h).astype(np.int64)
        off = int(-iu[iu < 0].sum())
        nb = int(np.abs(iu).sum()) + 1
        C = np.zeros((kmax + 1, nb), np.uint64)
        C[0, off] = 1
        for d in iu:
            d = int(d)
            N = C.copy()
            if d >= 0:
                N[1:, d:] += C[:-1, :nb - d]
            else:
                N[1:, :nb + d] += C[:-1, -d:]
            C = N
        res = []
        for k, T in zip(ks, Ts):
            t = int(math.ceil((T - TOL - J0) / h - 1e-9)) + off
            t = min(max(t, 0), nb)
            res.append(sum(int(C[j, t:].sum(dtype=np.uint64)) for j in range(int(k) + 1)))
        out[mode] = res
    return out["lo"], out["hi"]


def dev_seg(b):
    env, Js, mu, loader = dev_env_seg(b)
    eps = b["eps"]
    S, A = int(env.S), int(env.A)
    assert A == 2
    w = np.asarray(env.w, float)
    cost = np.asarray(env.problems.cost, np.int64)
    budgets = np.asarray(env.problems.budgets, np.int64)
    assert (cost[:, 0] == 0).all()
    J0 = float((w * mu[:, 0]).sum())
    u = w * (mu[:, 1] - mu[:, 0])
    c1 = np.unique(cost[:, 1])
    card = len(c1) == 1
    ks = [int(B // int(c1[0])) for B in budgets] if card else None
    Ts = [float(Js[q] - eps) for q in range(len(budgets))]
    h = eps * 1e-5
    per_q, tabs, nodes_tot, notes, xcheck = [], None, 0, [], []
    method = None
    br = card_dp_bracket(J0, u, ks, Ts, h) if card else None
    for q in range(len(budgets)):
        T, B = Ts[q], int(budgets[q])
        lo = hi = None
        if S <= 16:
            nf, ng, nbd = count_enum(w, mu, cost, B, T)
            method = "enumeration of all 2^S policies (exact)"
        elif S <= 32:
            nf, ng, nbd, tabs = count_mitm(w, mu, cost, B, T, tabs)
            method = "meet-in-the-middle over two 2^16 halves (exact)"
        else:
            assert card, "S = 64 counter needs uniform treatment costs (cardinality classes)"
            nf = sum(math.comb(S, j) for j in range(ks[q] + 1))
            nbd = None
            ng, nodes = count_card_bnb(J0, u, ks[q], T)
            nodes_tot += nodes
            lo, hi = br[0][q], br[1][q]
            if ng is None and lo == hi:
                ng = lo
            method = ("exact branch-and-bound over the cardinality structure; where its node limit is hit, the "
                      "rigorous rounding-DP bracket [lo, hi] (grid h = 1e-5 eps)")
            if ng is None:
                notes.append(f"q{q}: branch-and-bound node limit hit; bracket [{lo}, {hi}] of {nf}")
        if card and S <= 32:
            # the S = 64 counters must agree with / bracket the exact count on the enumerable classes
            xcheck.append(dict(q=q, exact=ng, dp_lo=br[0][q], dp_hi=br[1][q],
                               bracket_ok=bool(br[0][q] <= ng <= br[1][q])))
        if ng is not None:
            lo = hi = ng
        per_q.append(dict(n_feasible=int(nf), n_eps_opt=(None if ng is None else int(ng)),
                          n_eps_opt_lo=int(lo), n_eps_opt_hi=int(hi), share=(None if ng is None else ng / nf),
                          share_lo=lo / nf, share_hi=hi / nf, boundary_within_1e_9=nbd))
    exact = all(p["n_eps_opt"] is not None for p in per_q)
    nac = int(sum(J0 >= Js[q] - eps - TOL for q in range(len(budgets))))
    out = dict(S=S, A=A, n_policies=2 ** S, per_problem=per_q, J0=J0, Jstar=[float(x) for x in Js],
               n_allctrl=nac, dev_base_rate=float(np.asarray(env.outcomes_view("visit")).mean()),
               budgets=budgets.tolist(), max_treated=ks, treat_costs=sorted(set(cost[:, 1].tolist())),
               method=method, loader=loader, exact=exact, notes=notes, dp_bracket_cross_check=xcheck)
    if S > 32:
        out["bnb_nodes"] = nodes_tot
    nfs = sum(p["n_feasible"] for p in per_q)
    out["pooled_n_feasible"] = int(nfs)
    out["pooled_n_eps_opt_lo"] = int(sum(p["n_eps_opt_lo"] for p in per_q))
    out["pooled_n_eps_opt_hi"] = int(sum(p["n_eps_opt_hi"] for p in per_q))
    if exact:
        out["pooled_n_eps_opt"] = out["pooled_n_eps_opt_lo"]
    return out


# ------------------------------------------------------------------------------------------------- eval (sealed rows)
def rows_path(file):
    """The sealed row file as shipped: plain results.jsonl (authors' tree) or results.jsonl.gz (anonymous artifact,
    lock-v9 rows), as reproduce_frozen.sealed_rows_path accepts.  Returns None if neither exists."""
    p = FULL / file
    for c in (p, p.with_name(p.name + ".gz")):
        if c.exists():
            return c
    return None


def read_rows(file):
    p = rows_path(file)
    if p is None:
        raise FileNotFoundError(f"exp/results/full/{file}[.gz] not found")
    op = gzip.open if p.name.endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def eval_counts(src):
    rows = read_rows(src["file"])
    out = {}
    names = [("primary", src["primary"]), ("rival", src["rival"])]
    if src.get("rival_ni"):
        names.append(("rival_ni", src["rival_ni"]))
    for role, m in names:
        rr = [r for r in rows if r["method"] == m and abs(float(r["eps"]) - src["eps"]) < 1e-15]
        assert len(rr) == 200 and len({int(r["seed"]) for r in rr}) == 200, (src["file"], m, len(rr))
        tau = {int(r["tau_R"]) for r in rr}
        assert len(tau) == 1
        tau = tau.pop()
        if "n80_lt_tau" in rr[0]:
            not_before = sum(not bool(r["n80_lt_tau"]) for r in rr)
            hfield = "n80_lt_tau == False"
        else:
            not_before = sum(int(r["N80_pen"]) >= tau for r in rr)
            hfield = "N80_pen >= tau_R (no n80_lt_tau field in this lock)"
        if "exhaustion_at_k80" in rr[0]:
            exh = sum(float(r["exhaustion_at_k80"]["all"]) > 0 for r in rr)
            exh_ctrl_mean = float(np.mean([float(r["exhaustion_at_k80"]["ctrl"]) for r in rr]))
            efield = "exhaustion_at_k80['all'] > 0"
        else:
            exh, exh_ctrl_mean, efield = None, None, "not recorded in this lock's rows"
        out[role] = dict(method=m, n_streams=len(rr), tau_R=tau, n_not_12of15_before_tauR=int(not_before),
                         horizon_field=hfield, n_any_pool_exhausted_at_N80=exh,
                         mean_ctrl_exhausted_fraction_at_N80=exh_ctrl_mean, exhaustion_field=efield,
                         seeds=[min(int(r["seed"]) for r in rr), max(int(r["seed"]) for r in rr)])
    return dict(lock=src["lock"], source=f"exp/results/full/{src['file']}", eps=src["eps"],
                rival_basis=src["why"], analysis=f"exp/results/full/{src['analysis']}", **out)


# ------------------------------------------------------------------------------------------------- main
def fmt_share(x, nd=3):
    return "n.e." if x is None else f"{x:.{nd}f}"


def main():
    t0 = time.time()
    res = {"what": "r9 decision-difficulty summary (external review r9, P1-05); analysis of existing data only",
           "definitions": {
               "eps_optimal": "J*_q - J(pi) <= eps + 1e-12 on the DEV half's true cell means (J*_q by exact DP)",
               "a_pooled_share": "sum_q #eps-optimal feasible / sum_q #feasible (S12 census)",
               "b_mean_share": "mean over the 15 problems of #eps-optimal feasible / #feasible (v12 gate, "
                               "obd_v11_eval.eps_opt_share)",
               "c_all_control": "#problems with J*_q - J(all-control) <= eps (v12 gate criterion ii)",
               "eps_rel_base": "eps / mean outcome of the DEV half (all rows)",
               "eps_rel_headroom": "eps / (J*_15 - J(all-control)) on the DEV half (loosest budget)",
               "eval_not_before_tauR": "sealed EVAL rows: streams whose N80_pen is not strictly below tau_R",
               "eval_exhausted": "sealed EVAL rows: streams with any (segment, arm) pool exhausted at the N80 "
                                 "checkpoint (exhaustion_at_k80['all'] > 0); recorded from lock v10 on"},
           "blocks": [], "reproduction_checks": []}
    for b in BLOCKS:
        tb = time.time()
        d = dev_enum(b["layer"], b["eps"]) if b["kind"] == "enum" else dev_seg(b)
        nq = len(d["per_problem"])
        shares = [p["share"] for p in d["per_problem"]]
        d["mean_share"] = None if any(s is None for s in shares) else float(np.mean(shares))
        d["pooled_share"] = (d["pooled_n_eps_opt"] / d["pooled_n_feasible"]) if "pooled_n_eps_opt" in d else None
        if d["mean_share"] is None:          # rigorous bracket (S = 64 only)
            d["mean_share_lo"] = float(np.mean([p["share_lo"] for p in d["per_problem"]]))
            d["mean_share_hi"] = float(np.mean([p["share_hi"] for p in d["per_problem"]]))
            d["pooled_share_lo"] = d["pooled_n_eps_opt_lo"] / d["pooled_n_feasible"]
            d["pooled_share_hi"] = d["pooled_n_eps_opt_hi"] / d["pooled_n_feasible"]
        d["n_problems"] = nq
        d["eps_rel_base"] = b["eps"] / d["dev_base_rate"]
        d["eps_rel_headroom"] = b["eps"] / (d["Jstar"][-1] - d["J0"])
        d["seconds"] = round(time.time() - tb, 1)
        ev = [eval_counts(s) for s in EVAL_SRC[b["id"]]]
        res["blocks"].append(dict(id=b["id"], label=b["label"], locks=b["locks"], role=b["role"], eps=b["eps"],
                                  dev=d, eval=ev))
        # reproduction checks
        if b["id"] in PUBLISHED:
            p = PUBLISHED[b["id"]]
            for k in ("pooled_n_feasible", "pooled_n_eps_opt", "n_allctrl"):
                if k in p:
                    res["reproduction_checks"].append(dict(block=b["id"], quantity=k, published=p[k], computed=d[k],
                                                           match=(p[k] == d[k]), source=p["src"]))
            if "mean_share" in p:
                res["reproduction_checks"].append(dict(block=b["id"], quantity="mean_share", published=p["mean_share"],
                                                       computed=d["mean_share"],
                                                       match=abs(p["mean_share"] - d["mean_share"]) < 1e-12,
                                                       source=p["src"]))
        print(f"[{b['id']}] S={d['S']} pooled={share_str(d, 'pooled_share', 5)} mean={share_str(d, 'mean_share', 5)} "
              f"allctrl={d['n_allctrl']}/{nq} eps/base={d['eps_rel_base']:.3f} ({d['seconds']}s) "
              + " | ".join(f"{e['lock']}: P {e['primary']['n_not_12of15_before_tauR']} "
                           f"R {e['rival']['n_not_12of15_before_tauR']} exh P {e['primary']['n_any_pool_exhausted_at_N80']}"
                           f" R {e['rival']['n_any_pool_exhausted_at_N80']}" for e in ev), flush=True)
    res["all_reproduced"] = all(c["match"] for c in res["reproduction_checks"])
    res["seconds_total"] = round(time.time() - t0, 1)
    (IT / "writing/r9_difficulty.json").write_text(json.dumps(res, indent=1))
    write_tex(res)
    print(json.dumps(res["reproduction_checks"], indent=1))
    print("all_reproduced", res["all_reproduced"])


# ------------------------------------------------------------------------------------------------- LaTeX
def share_str(d, key, nd=3):
    """Exact value, or the rigorous bracket when it rounds to one value at nd decimals (marked *); else 'n.e.'."""
    if d.get(key) is not None:
        return f"{d[key]:.{nd}f}"
    lo, hi = d.get(key + "_lo"), d.get(key + "_hi")
    if lo is None:
        return "n.e."
    a, b = f"{lo:.{nd}f}", f"{hi:.{nd}f}"
    return f"{a}*" if a == b else "n.e."


def _pct(x):
    return f"{100 * x:.1f}"


def _eps_tex(e):
    m = {0.001: "0.001", 0.02: "0.02", 0.03: "0.03", 0.04: "0.04", 0.05: "0.05", 0.004: "0.004", 0.006: "0.006",
         0.008: "0.008", 3e-4: r"$3{\cdot}10^{-4}$", 7.5e-4: r"$7.5{\cdot}10^{-4}$"}
    return m.get(e, f"{e:g}")


def _hz(e, key="n_not_12of15_before_tauR"):
    return f"{e['primary'][key]}/{e['rival'][key]}"


def _exh(e):
    p, r = e["primary"]["n_any_pool_exhausted_at_N80"], e["rival"]["n_any_pool_exhausted_at_N80"]
    return "n.r." if p is None else f"{p}/{r}"


def _ac(d):
    return f"{d['n_allctrl']}"


EPS_PAIR = {("LE-32", "LE-64"): "0.006/.008", ("OBD-W", "OBD-M"): r"$7.5/10{\cdot}10^{-4}$"}


def write_tex(res):
    B = {b["id"]: b for b in res["blocks"]}
    short = {b["id"]: b["short"] for b in BLOCKS}
    # ---------------- supplement: full table
    L = [r"% generated by writing/scripts/difficulty_r9.py -- do not edit by hand",
         r"\begin{tabular}{@{}lllrrrrrll@{}}", r"\toprule",
         r" & & & \multicolumn{4}{c}{\emph{Development half (dev truth)}} & & \multicolumn{2}{c}{\emph{Sealed "
         r"evaluation rows}} \\",
         r"\cmidrule(lr){4-7}\cmidrule(l){9-10}",
         r"Block & Lock & $\eps$ & $\eps$/base & $\eps$/head & Pooled & Mean & Ctrl & Hor.\ P/R & Exh.\ P/R \\",
         r"\midrule"]
    for bid, b in B.items():
        d = b["dev"]
        for j, e in enumerate(b["eval"]):
            if j == 0:
                pooled = share_str(d, "pooled_share")
                if d["pooled_share"] is not None and d["S"] <= 9:
                    pooled += f" ({d['pooled_n_eps_opt']:,}/{d['pooled_n_feasible']:,})".replace(",", "{,}")
                cells = [short[bid], e["lock"], _eps_tex(b["eps"]), _pct(d["eps_rel_base"]) + r"\%",
                         f"{d['eps_rel_headroom']:.2f}", pooled, share_str(d, "mean_share"), _ac(d)]
            else:
                cells = ["", e["lock"], "", "", "", "", "", ""]
            hz = _hz(e)
            if e.get("rival_ni"):
                hz += f" (NI {e['rival_ni']['n_not_12of15_before_tauR']})"
            cells += [hz, _exh(e)]
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    p = IT / "writing/supplement/r9_tables/difficulty.tex"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    # ---------------- main paper: 9 rows x 6 columns; X5 S = 16/32/64 share one row (identical dev difficulty)
    M = [r"% generated by writing/scripts/difficulty_r9.py -- do not edit by hand",
         r"% dev = development half, dev truth; Hor./Exh. = sealed evaluation rows (P = primary, R = governing rival)",
         r"\small", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}{@{}llrrll@{}}", r"\toprule",
         r"Log, $S$ & $\eps$ (\% base) & Share & Ctrl & Hor.\ P/R & Exh.\ P/R \\", r"\midrule"]

    def e_short(x):
        return {0.001: ".001", 0.02: ".02", 0.03: ".03", 0.04: ".04", 0.05: ".05", 0.004: ".004", 0.006: ".006",
                0.008: ".008", 3e-4: ".0003", 7.5e-4: ".00075"}[x]

    def one(lbl, bid):
        b, d = B[bid], B[bid]["dev"]
        e = b["eval"][-1]
        return (f"{lbl} & {e_short(b['eps'])} ({100 * d['eps_rel_base']:.1f}) & {share_str(d, 'mean_share', 2)} & "
                f"{_ac(d)} & {_hz(e)} & {_exh(e)} \\\\")

    M.append(one("Criteo, 9", "CR9"))
    M.append(one("X5, 9", "X9"))
    ids = ["X5-16", "X5-32", "X5-64"]
    ds, es = [B[i]["dev"] for i in ids], [B[i]["eval"][-1] for i in ids]
    sh = sorted({share_str(d, "mean_share", 2) for d in ds})
    ac = sorted({_ac(d) for d in ds})
    assert len(sh) == 1 and len(ac) == 1, "X5 S=16/32/64 no longer share one dev difficulty; split the row"
    hp = [e["primary"]["n_not_12of15_before_tauR"] for e in es]
    hr = [e["rival"]["n_not_12of15_before_tauR"] for e in es]
    xp = [e["primary"]["n_any_pool_exhausted_at_N80"] for e in es]
    xr = [e["rival"]["n_any_pool_exhausted_at_N80"] for e in es]
    rng = lambda v: f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}"
    M.append(f"X5, 16--64$^\\ddagger$ & .03--.05 ({100 * ds[0]['eps_rel_base']:.1f}--{100 * ds[-1]['eps_rel_base']:.1f})"
             f" & {sh[0]} & {ac[0]} & {rng(hp)}/{rng(hr)} & {rng(xp)}/{rng(xr)} \\\\")
    M.append(one("Lenta, 16", "LE-16"))
    M.append(one("Lenta, 32", "LE-32"))
    M.append(one("Lenta, 64", "LE-64"))
    M.append(one("OBD all, 9", "OBD"))
    M.append(one("OBD women, 8$^\\dagger$", "OBD-W"))
    M.append(one("OBD men, 10$^\\dagger$", "OBD-M"))
    M += [r"\bottomrule", r"\end{tabular}"]
    (IT / "writing/latex_acm/r9_difficulty_main.tex").write_text("\n".join(M) + "\n")


def check_eval():
    """--check-eval: recompute ONLY the EVAL counts (Hor./Exh. and the other per-stream counts of every registered
    sealed source) from the row files present (plain or .gz) and compare each entry, as a whole, with the stored
    writing/r9_difficulty.json.  Nothing is written.  Exit status 0 iff every registered source is present and every
    entry is identical; 1 otherwise."""
    stored = J(IT / "writing/r9_difficulty.json")
    sb = {b["id"]: b for b in stored["blocks"]}
    n_ok, bad = 0, []
    for b in BLOCKS:
        st = sb.get(b["id"], {}).get("eval")
        srcs = EVAL_SRC[b["id"]]
        if st is None or len(st) != len(srcs):
            bad.append(f"{b['id']}: stored json has {None if st is None else len(st)} eval entries, registry "
                       f"{len(srcs)}")
            continue
        for src, old in zip(srcs, st):
            p = rows_path(src["file"])
            if p is None:
                bad.append(f"{b['id']} {src['lock']}: row file exp/results/full/{src['file']}[.gz] missing")
                continue
            try:
                new = json.loads(json.dumps(eval_counts(src)))
            except AssertionError as e:
                bad.append(f"{b['id']} {src['lock']}: row file failed the count assertions {e}")
                continue
            diff = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
            tag = (f"{b['id']:6s} {src['lock']:4s} {str(p.relative_to(IT)):58s} Hor P/R "
                   f"{new['primary']['n_not_12of15_before_tauR']}/{new['rival']['n_not_12of15_before_tauR']}"
                   + (f" NI {new['rival_ni']['n_not_12of15_before_tauR']}" if "rival_ni" in new else "")
                   + f"  Exh P/R {new['primary']['n_any_pool_exhausted_at_N80']}/"
                     f"{new['rival']['n_any_pool_exhausted_at_N80']}")
            if diff:
                bad.append(f"{tag}  DIFFERS in {diff}")
                print("DIFF", tag, diff)
            else:
                n_ok += 1
                print("ok  ", tag)
    print(f"check-eval: {n_ok} of {sum(len(v) for v in EVAL_SRC.values())} registered sealed sources recomputed "
          f"identical to writing/r9_difficulty.json; {len(bad)} problem(s)")
    for x in bad:
        print("  -", x)
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--check-eval" in sys.argv:    # recompute only the EVAL counts from the row files present; compare; no writes
        sys.exit(check_eval())
    if "--tex-only" in sys.argv:      # re-render the LaTeX fragments from the written json (no recomputation)
        write_tex(J(IT / "writing/r9_difficulty.json"))
    else:
        main()
