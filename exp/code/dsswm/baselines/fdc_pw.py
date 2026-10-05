"""FDC-PW: FDC-DP with a prior-weighted (shell-localised) union ledger (NEW file, v11 workstream; plan/fdc_pw_theory.md).

FDC-DP's union is over the index set I = {(q, pi, k) : pi in Pi_q}; the event for (q, pi, k) concerns the direction
(pi*_q, pi) and is used when pi is the certificate's centre.  Any frozen weights w_q(pi) >= 0 with sum_pi w_q(pi) <= 1
may replace the uniform allocation: give index (q, pi, k) the level delta_main * w_q(pi) / (Q K), i.e. the exponent

    beta(q, pi) = ln(Q K / delta_main) + ln(1 / w_q(pi)).

The union bound still sums to delta_main.  Only the CENTRE's exponent enters a certificate (the challenger maximum is
unchanged), so the knapsack DP / branch-and-bound of FDC-DP apply verbatim with const = beta(q, centre) / lambda.
Because the event holds simultaneously for every index, a certificate may be attempted at several data-dependent
candidate centres; each attempt that passes is epsilon-good on the event.

Weights (pre-stated, outcome-free): a reference policy ref_q (the knapsack argmax of a frozen covariate-only score per
segment), Hamming shells d(pi) = #{s : pi(s) != ref_q(s)}, shell counts N_q(d) from a counting DP over
(segment, cost, d), and a shell mixture rho_d:

    w_q(pi) = rho_{d(pi)} / N_q(d(pi)),   rho_d = 0.5 * 2^{-(d+1)} / (1 - 2^{-(S+1)}) + 0.5 / (S + 1),

so sum_pi w_q(pi) = sum_d rho_d = 1 and the worst-case penalty against a uniform ledger restricted to Pi_q is at most
ln(2 (S + 1)).  Counts are float64 and inflated by a relative 1e-9 (an over-count only raises beta: conservative).

Candidate centres per problem (centres="ball", default): the empirical argmax restricted to the Hamming ball of radius
r around ref_q, for r in RADII (0 = ref_q itself; r = S = FDC-DP's centre), computed by one knapsack DP with a
flip-count state.  Each candidate's certificate is evaluated with scheme (a); scheme (b) is then run once, on the
candidate with the smallest U_a.  centres="both" keeps the two-candidate rule {argmax, ref_q}.
"""
from __future__ import annotations

import math

import numpy as np

from .fdc_dp import (FDCDP, FDCDPTimeUniform, bnb, cell_tables, dp_argmax, joint_columns, lambda_grid,
                     reduce_costs, scheme_a_U)

COUNT_INFLATE = 1.0 + 1e-9
RADII = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
_NEG = -np.inf


def dp_argmax_ball(vals, cost, budget, ref, radii):
    """For each r in radii: argmax_{pi feasible, Hamming(pi, ref) <= r} sum_s vals[s, pi(s)] (exact DP with a
    remaining-flips state; ties -> lowest arm).  Returns {r: pi or None}."""
    vals = np.asarray(vals, dtype=float)
    cost, (budget,), _ = reduce_costs(cost, [budget])
    S, A = vals.shape
    cap = int(budget)
    out = {}
    if cap < 0:
        return {r: None for r in radii}
    R = S
    best = np.full((S + 1, cap + 1, R + 1), _NEG)
    best[S] = 0.0
    for s in range(S - 1, -1, -1):
        row = np.full((cap + 1, R + 1), _NEG)
        for a in range(A):
            k = int(cost[s, a])
            if k > cap:
                continue
            cand = np.full((cap + 1, R + 1), _NEG)
            if a == int(ref[s]):
                cand[k:, :] = best[s + 1, :cap + 1 - k, :] + vals[s, a]
            else:
                cand[k:, 1:] = best[s + 1, :cap + 1 - k, :-1] + vals[s, a]
            row = np.maximum(row, cand)
        best[s] = row
    for r in radii:
        r = min(int(r), R)
        if not np.isfinite(best[0, cap, r]):
            out[r] = None
            continue
        pi = np.zeros(S, dtype=np.int64)
        c, rr = cap, r
        for s in range(S):
            tgt = best[s, c, rr]
            for a in range(A):
                k = int(cost[s, a])
                if k > c:
                    continue
                flip = int(a != int(ref[s]))
                if flip > rr:
                    continue
                v = best[s + 1, c - k, rr - flip] + vals[s, a]
                if v >= tgt - 1e-12 * max(1.0, abs(tgt)):
                    pi[s] = a
                    c -= k
                    rr -= flip
                    break
        out[r] = pi
    return out


def shell_rho(S):
    d = np.arange(S + 1, dtype=float)
    geo = 0.5 * 2.0 ** (-(d + 1)) / (1.0 - 2.0 ** (-(S + 1)))
    return geo + 0.5 / (S + 1)


def shell_counts(cost, cap, ref):
    """N(d) = #{pi : sum_s cost[s, pi(s)] <= cap, Hamming(pi, ref) = d}, float64 (counting DP over cost x d)."""
    cost = np.asarray(cost, dtype=np.int64)
    S, A = cost.shape
    cap = int(cap)
    if cap < 0:
        return np.zeros(S + 1)
    N = np.zeros((cap + 1, S + 1))
    N[0, 0] = 1.0
    for s in range(S):
        nxt = np.zeros_like(N)
        for a in range(A):
            c = int(cost[s, a])
            if c > cap:
                continue
            dd = 0 if a == int(ref[s]) else 1
            if dd:
                nxt[c:, 1:] += N[:cap + 1 - c, :-1]
            else:
                nxt[c:, :] += N[:cap + 1 - c, :]
        N = nxt
    return N.sum(0) * COUNT_INFLATE


class _PWMixin:
    """Shared ledger / centre logic for FDC-PW and TU-FDC-PW."""

    def _pw_init(self, ref_vals, centres):
        self.ref_vals = np.asarray(ref_vals, dtype=float)
        if centres not in ("ball", "both", "argmax", "ref"):
            raise ValueError(centres)
        self.centres = centres

    def _pw_setup(self, ctx):
        sp = ctx.sp
        if self.ref_vals.shape != (ctx.S, ctx.A):
            raise ValueError("ref_vals must have shape (S, A)")
        cost, budgets, g = reduce_costs(sp.cost, sp.budgets)
        K = int(self.ledger["K"]) if "K" in self.ledger else int(len(self._grid_set))
        dm = self.split[0]
        base = math.log(ctx.Q * K / dm)
        rho = shell_rho(ctx.S)
        self.pw = []
        for q in range(ctx.Q):
            _, ref = dp_argmax(self.ref_vals, sp.cost, int(sp.budgets[q]))
            if ref is None:
                raise RuntimeError(f"problem {q}: empty policy class")
            ref = np.asarray(ref, dtype=np.int64)
            Nd = shell_counts(cost, int(budgets[q]), ref)
            with np.errstate(divide="ignore"):
                beta_d = base + np.log(np.where(Nd > 0, Nd, np.inf)) - np.log(rho)
            self.pw.append({"ref": ref, "N_d": Nd, "beta_d": beta_d})
        bmin = min(float(np.min(p["beta_d"][np.isfinite(p["beta_d"])])) for p in self.pw)
        bmax = max(float(np.max(p["beta_d"][np.isfinite(p["beta_d"])])) for p in self.pw)
        self.ledger.update({"pw_base": base, "pw_beta_min": bmin, "pw_beta_max": bmax,
                            "pw_rule": "w_q(pi) = rho_d / N_q(d), d = Hamming to ref_q; rho = 0.5 geometric + 0.5 "
                                       "uniform over d = 0..S", "pw_centres": self.centres,
                            "pw_ref_beta": [float(p["beta_d"][0]) for p in self.pw]})
        self._bminmax = (bmin, bmax)

    def beta_of(self, q, pi):
        p = self.pw[q]
        d = int(np.sum(np.asarray(pi) != p["ref"]))
        return float(p["beta_d"][d]), d

    def _pw_lam(self, ctx, n, N, lo, hi):
        bmin, bmax = self._bminmax
        l1 = lambda_grid(bmin, ctx.eps, ctx.w, n, N, lo, hi, self.grid_ratio)
        l2 = lambda_grid(bmax, ctx.eps, ctx.w, n, N, lo, hi, self.grid_ratio)
        return np.unique(np.concatenate([l1, l2]))

    def certify_with(self, ctx, mu, n, N, lo, hi, undecided=None, rect=None, t=None, tables=None):
        import time
        if rect is not None:
            raise NotImplementedError("FDC-PW: hybrid rectangle column not supported")
        t0 = time.perf_counter()
        sp = ctx.sp
        cost, budgets, _ = reduce_costs(sp.cost, sp.budgets)
        if tables is None:
            lam = self._pw_lam(ctx, n, N, lo, hi)
            P, det = cell_tables(ctx.w, n, N, lo, hi, lam)
        else:
            lam, P, det = tables
        vals = np.asarray(ctx.w, float)[:, None] * np.asarray(mu, float)
        out, detail = [None] * ctx.Q, [None] * ctx.Q
        st = {"t": t, "nodes": 0, "bnb_calls": 0, "node_limit_hits": 0, "a_certified": 0, "b_certified": 0,
              "ref_certified": 0, "argmax_certified": 0, "ball_certified": 0}
        cache = {}
        for q in range(ctx.Q):
            _, ph = dp_argmax(vals, sp.cost, int(sp.budgets[q]))
            if ph is None:
                raise RuntimeError(f"problem {q}: empty policy class")
            cands = []
            if self.centres == "ball":
                balls = dp_argmax_ball(vals, sp.cost, int(sp.budgets[q]), self.pw[q]["ref"], RADII)
                seen = set()
                for r in sorted(balls):
                    pi = balls[r]
                    if pi is None:
                        continue
                    k_ = tuple(int(x) for x in pi)
                    if k_ not in seen:
                        seen.add(k_)
                        cands.append((f"r{r}", k_))
            if self.centres in ("both", "argmax"):
                cands.append(("argmax", tuple(int(x) for x in ph)))
            if self.centres in ("both", "ref"):
                rk = tuple(int(x) for x in self.pw[q]["ref"])
                if rk not in [c[1] for c in cands]:
                    cands.append(("ref", rk))
            if undecided is not None and not undecided[q]:
                out[q] = (False, cands[0][1], float("nan"))
                continue
            best, d_q = None, {"q": q, "tries": []}
            cap = int(budgets[q])
            if self.centres == "ball":
                best = self._ball_certify(ctx, q, cands, cap, cost, mu, P, det, lam, cache, st, d_q)
                cands = []
            for label, key in cands:
                beta, dist = self.beta_of(q, key)
                ck = (key, round(beta, 12))
                if ck not in cache:
                    cache[ck] = joint_columns(ctx.w, mu, np.asarray(key, dtype=np.int64), P, det, lam, beta)
                const, T, diff = cache[ck]
                U = max(0.0, scheme_a_U(const, T, cost, diff, [cap])[0])
                ok, how = U <= ctx.eps, "a"
                if not ok and self.scheme == "b":
                    r = bnb(const, T, cost, diff, cap, thr=ctx.eps, node_limit=self.node_limit)
                    st["nodes"] += r["nodes"]
                    st["bnb_calls"] += 1
                    if r["status"] == "node_limit":
                        st["node_limit_hits"] += 1
                    elif r["status"] == "certified":
                        ok, how = True, "b"
                d_q["tries"].append({"centre": label, "key": key, "d": dist, "beta": beta, "U_a": U, "ok": bool(ok),
                                     "how": how})
                if ok:
                    best = (label, key, U, how)
                    break
            if best is None:
                tr = min(d_q["tries"], key=lambda x: x["U_a"])
                out[q] = (False, tr["key"], float(tr["U_a"]))
            else:
                lab, key, U, how = best
                lab = "ref" if lab in ("ref", "r0") else ("argmax" if lab == "argmax" or lab == f"r{max(RADII)}"
                                                           else "ball")
                st[f"{lab}_certified"] = st.get(f"{lab}_certified", 0) + 1
                st[f"{how}_certified"] += 1
                out[q] = (True, key, float(U))
            detail[q] = d_q
        st["sec"] = round(time.perf_counter() - t0, 4)
        self.cert_stats.append(st)
        self.last_detail = detail
        return out


def _ball_certify(self, ctx, q, cands, cap, cost, mu, P, det, lam, cache, st, d_q):
    """Scheme (a) at every candidate (smallest beta first); if none passes and scheme == 'b', branch-and-bound on the
    candidate with the smallest U_a.  Returns (label, key, U, how) or None."""
    tries = []
    for label, key in cands:
        beta, dist = self.beta_of(q, key)
        ck = (key, round(beta, 12))
        if ck not in cache:
            cache[ck] = joint_columns(ctx.w, mu, np.asarray(key, dtype=np.int64), P, det, lam, beta)
        const, T, diff = cache[ck]
        U = max(0.0, scheme_a_U(const, T, cost, diff, [cap])[0])
        rec = {"centre": label, "key": key, "d": dist, "beta": beta, "U_a": U, "ok": U <= ctx.eps, "how": "a"}
        d_q["tries"].append(rec)
        tries.append((rec, ck))
        if rec["ok"]:
            return label, key, U, "a"
    if self.scheme == "b" and tries:
        rec, ck = min(tries, key=lambda x: x[0]["U_a"])
        const, T, diff = cache[ck]
        r = bnb(const, T, cost, diff, cap, thr=ctx.eps, node_limit=self.node_limit)
        st["nodes"] += r["nodes"]
        st["bnb_calls"] += 1
        if r["status"] == "node_limit":
            st["node_limit_hits"] += 1
        elif r["status"] == "certified":
            rec["ok"], rec["how"] = True, "b"
            return rec["centre"], rec["key"], rec["U_a"], "b"
    return None


_PWMixin._ball_certify = _ball_certify


class FDCPW(_PWMixin, FDCDP):
    """FDC-PW(scheme): checkpoint-valid (K-grid) prior-weighted FDC-DP."""

    def __init__(self, ref_vals, centres="ball", scheme="b", **kw):
        kw.setdefault("name", None)
        name = kw.pop("name")
        super().__init__(scheme=scheme, **kw)
        self._pw_init(ref_vals, centres)
        self.name = name or f"FDC-PW({scheme})"

    def setup(self, ctx):
        super().setup(ctx)
        self.ledger["K"] = len(self._grid_set)
        self._pw_setup(ctx)

    def describe(self):
        d = super().describe()
        d["guarantee"] = "FWER <= 0.05 at the K checkpoints (plan/fdc_pw_theory.md Theorem PW-1)"
        return d


class FDCPWTimeUniform(_PWMixin, FDCDPTimeUniform):
    """TU-FDC-PW: Lemma TU reading of FDC-PW (widths frozen at block points, Delta_hat current)."""

    def __init__(self, block_points, ref_vals, centres="ball", scheme="b", **kw):
        kw.setdefault("name", None)
        name = kw.pop("name")
        super().__init__(block_points, scheme=scheme, **kw)
        self._pw_init(ref_vals, centres)
        self.name = name or f"TU-FDC-PW({scheme})"

    def setup(self, ctx):
        super().setup(ctx)
        self.ledger["K"] = len(self.block_points)
        self._pw_setup(ctx)

    def certify(self, ctx, st):
        if st.t < self._last_t:
            raise RuntimeError("evaluation times must be increasing")
        self._last_t = st.t
        while self._next_block < len(self.block_points) and self.block_points[self._next_block] <= st.t:
            if int(self.block_points[self._next_block]) != st.t:
                raise RuntimeError("block grid must be a subset of the evaluation grid")
            lo, hi = self._box(st)
            lam = self._pw_lam(ctx, st.n, st.N, lo, hi)
            P, det = cell_tables(ctx.w, st.n, st.N, lo, hi, lam)
            self._snap = (np.asarray(st.n).copy(), np.asarray(st.N).copy(), lo, hi, (lam, P, det))
            self._next_block += 1
        if self._snap is None:
            return [(False, None, float("inf")) for _ in range(ctx.Q)]
        n_k, N_k, lo, hi, tables = self._snap
        if int(self.block_points[self._next_block - 1]) != st.t:
            self.n_stale_evals += 1
        return self.certify_with(ctx, st.mu_hat, n_k, N_k, lo, hi, getattr(st, "undecided", None), t=st.t,
                                 tables=tables)

    def describe(self):
        d = super().describe()
        d["guarantee"] = "FWER <= 0.05, time-uniform on the frozen schedule (Lemma TU + Theorem PW-1)"
        return d
