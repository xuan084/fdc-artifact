"""Five-level causal reuse switch (methodology section 3, round 3). Learner side: no truth access.

Arms (what evidence is available on problem k of a stream):
  off      n0 + new data of problem k only; data of problems 0..k-1 is discarded (still billed when collected)
  vol      n0 + new data of problem k + SHADOW LEDGER: data that the same method accumulated on problems 0..k-1 of the
           same stream on a SIBLING platform (same theta*, independent noise seed = stream noise seed + 7919),
           truncated / padded to exactly N_k rows, N_k = number of real old-problem rows this arm has collected so far.
           Shadow rows are tagged provenance='replay_vol' (padding: 'replay_vol_pad') and are never billed.
  orth     n0 + new data of problem k + N_k sibling rows from a designed occupancy xi_perp (Frank-Wolfe, built by
           evidence.orth_design in r3_p3). Here only the HOOK: `orth_provider(k, pid, n_rows, context)` returns a list
           of authentic sibling observations or None (-> 'orth_infeasible', the problem is skipped, never downgraded).
  ev(m)    n0 + full old ledger + new data, but certification of problem k is allowed only after >= m new steps
  full     n0 + full old ledger, certification with 0 new steps allowed

Billing invariants (asserted for every arm after every problem, `check_billing`):
  env.n_steps            == lr.n_rounds_billed      (every real platform round, n0 included, is billed once)
  lr.n_rounds_total      == lr.n_steps + lr.n_replay (rows inside the active evidence set = real rows + replay rows)
  lr.inner.n_rounds      == lr.n_rounds_total        (the confidence set has seen exactly those rows)
and per arm: off -> n_replay == 0 and n_steps == n0 + new_k; vol/orth -> n_replay == N_k; ev/full -> n_steps == billed.
Replay data is real env.step() output of a sibling platform (provenance-authenticated), NOT model rollouts. It is an
oracle 'free data' control, not a deployable method.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.provenance import require_authentic

ARM_BASES = ("off", "vol", "orth", "ev", "full")
SHADOW_SEED_OFFSET = 7919
PROV_INITIAL = "initial"
PROV_REAL = "real"
PROV_VOL = "replay_vol"
PROV_VOL_PAD = "replay_vol_pad"
PROV_ORTH = "replay_orth"
REPLAY_PROVS = (PROV_VOL, PROV_VOL_PAD, PROV_ORTH)
TAUS = (1500, 3000, 6000)
T_MAX = 6000


class BillingError(AssertionError):
    pass


def parse_arm(arm: str) -> tuple[str, int]:
    """'off' | 'vol' | 'orth' | 'full' | 'ev1' | 'ev20' | 'ev(200)' -> (base, m). m = min new steps before certifying."""
    a = arm.strip().lower()
    if a in ("off", "vol", "orth", "full"):
        return a, 0
    mt = re.fullmatch(r"ev\(?(\d+)\)?", a)
    if mt:
        m = int(mt.group(1))
        if m < 1:
            raise ValueError("ev(m) needs m >= 1")
        return "ev", m
    raise ValueError(f"unknown reuse arm {arm!r}")


def shadow_noise_seed(stream_noise_seed: int) -> int:
    return int(stream_noise_seed) + SHADOW_SEED_OFFSET


@dataclass
class ProvRow:
    serial: int
    provenance: str
    problem_id: str | None      # problem whose new data this row was (for replay: the sibling problem id)
    active_pid: str | None      # problem whose evidence set the row was fed into


class SwitchedEvidence:
    """The evidence set used on ONE problem (or the persistent one for ev/full). Wraps an inner confidence set built
    by `make_set()` (SeqLRSet API: update / mask / mle / log_ratio / size / cutoff / n_rounds) and keeps a provenance
    label for every row it contains."""

    def __init__(self, make_set, switch: "ReuseSwitch"):
        self.inner = make_set()
        self._switch = switch
        self.rows: list[ProvRow] = []
        self.obs: list = []
        self.n_steps = 0         # real (billed, this platform) rows inside this set, n0 included
        self.n_replay = 0        # replay rows inside this set (never billed)

    # --- feeding
    def _feed(self, obs, prov: str, pid, active_pid):
        require_authentic(obs)
        self.inner.update(obs)
        self.rows.append(ProvRow(int(obs.serial), prov, pid, active_pid))
        self.obs.append(obs)
        if prov in REPLAY_PROVS:
            self.n_replay += 1
        else:
            self.n_steps += 1

    # --- accounting
    @property
    def n_rounds_total(self) -> int:
        return len(self.rows)

    @property
    def n_rounds_billed(self) -> int:
        return self._switch.n_rounds_billed

    def provenance_counts(self) -> dict:
        out: dict = {}
        for r in self.rows:
            out[r.provenance] = out.get(r.provenance, 0) + 1
        return out

    # --- delegate the confidence-set API
    def mask(self):
        return self.inner.mask()

    def mle(self):
        return self.inner.mle()

    def log_ratio(self):
        return self.inner.log_ratio()

    def size(self):
        return self.inner.size()

    def cutoff(self):
        return self.inner.cutoff()

    def statistic(self):
        return self.inner.statistic()


class ShadowLedger:
    """Sibling-platform data for the `vol` arm. `sibling_rows` = [(obs, problem_id)] produced by the same method on
    the sibling platform, in collection order, for the problems of the same stream (initial rows excluded).
    `pad_fn(n)` must return >= n further authentic sibling observations (same sibling platform, e.g. uniform random
    legal actions after the sibling stream ended); the pool is drawn sequentially and reused as a prefix, so the
    shadow ledger for problem k is a deterministic function of (k, N_k)."""

    def __init__(self, sibling_rows, problem_order, pad_fn=None, sibling_env_steps: int | None = None):
        self.rows = [(o, pid) for o, pid in sibling_rows]
        for o, _ in self.rows:
            require_authentic(o)
        self.order = list(problem_order)
        self.pad_fn = pad_fn
        self.pad_pool: list = []
        self.sibling_env_steps = sibling_env_steps
        self.n_padded_max = 0

    def prefix(self, k: int) -> list:
        prev = set(self.order[:k])
        return [(o, pid) for o, pid in self.rows if pid in prev]

    def take(self, k: int, n_rows: int) -> list:
        """[(obs, provenance, sibling_pid)] of exactly n_rows rows for problem index k."""
        pre = self.prefix(k)
        out = [(o, PROV_VOL, pid) for o, pid in pre[:n_rows]]
        need = n_rows - len(out)
        if need > 0:
            if self.pad_fn is None:
                raise BillingError(f"shadow ledger short by {need} rows and no pad_fn given")
            while len(self.pad_pool) < need:
                new = list(self.pad_fn(need - len(self.pad_pool)))
                if not new:
                    raise BillingError("pad_fn returned no observations")
                for o in new:
                    require_authentic(o)
                self.pad_pool.extend(new)
            out += [(o, PROV_VOL_PAD, None) for o in self.pad_pool[:need]]
            self.n_padded_max = max(self.n_padded_max, need)
        assert len(out) == n_rows
        return out


@dataclass
class ProblemAccount:
    k: int
    pid: str
    arm: str
    new_env_steps: int = 0
    replay_steps: int = 0
    replay_pad_steps: int = 0
    n_rounds_total: int = 0
    n_rounds_billed: int = 0
    orth_feasible: bool | None = None
    prov_counts: dict = field(default_factory=dict)


class ReuseSwitch:
    """Controls which evidence each problem of a stream sees, and enforces billing.

    Usage (per stream, one platform):
        sw = ReuseSwitch('vol', make_set, init_obs, shadow=ShadowLedger(...))
        for k, q in enumerate(problems):
            lr = sw.begin_problem(k, q.pid)          # SwitchedEvidence for this problem
            if lr is None: ...                       # orth_infeasible
            while not done:
                if sw.may_certify() and certified(lr): break
                obs = handle.step(a); sw.record(obs)
            acc = sw.end_problem(handle.n_steps)     # billing assertions
    """

    def __init__(self, arm: str, make_set, init_obs, shadow: ShadowLedger | None = None, orth_provider=None):
        self.arm = arm
        self.base, self.m = parse_arm(arm)
        self.make_set = make_set
        self.init_obs = list(init_obs)
        for o in self.init_obs:
            require_authentic(o)
        self.n0 = len(self.init_obs)
        self.shadow = shadow
        self.orth_provider = orth_provider
        if self.base == "vol" and shadow is None:
            raise ValueError("arm 'vol' needs a ShadowLedger")
        self.n_rounds_billed = 0
        self.real_ledger: list = []          # [(obs, problem_id or None)] every billed real row in order
        self.cost_by_problem: dict = {}
        self.replay_by_problem: dict = {}
        self.accounts: list[ProblemAccount] = []
        self._k = None
        self._pid = None
        self._new = 0
        self._persistent = None
        self.active: SwitchedEvidence | None = None
        for o in self.init_obs:
            self._bill(o, None)
        if self.base in ("ev", "full"):
            self._persistent = SwitchedEvidence(make_set, self)
            for o in self.init_obs:
                self._persistent._feed(o, PROV_INITIAL, None, None)

    # --------------------------------------------------------------------------------------------- internals
    def _bill(self, obs, pid):
        require_authentic(obs)
        self.n_rounds_billed += 1
        self.real_ledger.append((obs, pid))
        if pid is not None:
            self.cost_by_problem[pid] = self.cost_by_problem.get(pid, 0) + 1

    @property
    def n_old_real(self) -> int:
        """N_k: real rows collected on earlier problems of this stream (n0 excluded)."""
        return self.n_rounds_billed - self.n0 - self._new

    # --------------------------------------------------------------------------------------------- API
    def begin_problem(self, k: int, pid: str, context=None) -> SwitchedEvidence | None:
        if self._k is not None:
            raise RuntimeError("end_problem() was not called for the previous problem")
        self._k, self._pid, self._new = k, pid, 0
        acc = ProblemAccount(k=k, pid=pid, arm=self.arm)
        self.accounts.append(acc)
        if self.base in ("ev", "full"):
            self.active = self._persistent
            return self.active
        ev = SwitchedEvidence(self.make_set, self)
        for o in self.init_obs:
            ev._feed(o, PROV_INITIAL, None, pid)
        n_k = self.n_old_real
        if self.base == "vol":
            for o, prov, spid in self.shadow.take(k, n_k):
                ev._feed(o, prov, spid, pid)
        elif self.base == "orth":
            rows = None if self.orth_provider is None else self.orth_provider(k, pid, n_k, context)
            if rows is None:
                acc.orth_feasible = False
                self.active = None
                return None
            rows = list(rows)
            if len(rows) != n_k:
                raise BillingError(f"orth provider returned {len(rows)} rows, need N_k = {n_k}")
            for o in rows:
                ev._feed(o, PROV_ORTH, None, pid)
            acc.orth_feasible = True
        self.replay_by_problem[pid] = ev.n_replay
        self.active = ev
        return ev

    def record(self, obs) -> None:
        """A NEW real platform round collected for the current problem (billed)."""
        if self._k is None:
            raise RuntimeError("record() outside a problem")
        self._bill(obs, self._pid)
        self._new += 1
        if self.active is not None:
            self.active._feed(obs, PROV_REAL, self._pid, self._pid)

    def may_certify(self) -> bool:
        if self.base == "ev":
            return self._new >= self.m
        return True

    @property
    def new_steps(self) -> int:
        return self._new

    def check_billing(self, env_n_steps: int) -> None:
        lr = self.active
        if env_n_steps != self.n_rounds_billed:
            raise BillingError(f"env.n_steps={env_n_steps} != lr.n_rounds_billed={self.n_rounds_billed}")
        if lr is None:
            return
        if lr.n_rounds_total != lr.n_steps + lr.n_replay:
            raise BillingError("n_rounds_total != n_steps + n_replay")
        if getattr(lr.inner, "n_rounds", lr.n_rounds_total) != lr.n_rounds_total:
            raise BillingError(f"inner set saw {lr.inner.n_rounds} rows, provenance says {lr.n_rounds_total}")
        n_k = self.n_old_real
        if self.base == "off":
            ok = lr.n_replay == 0 and lr.n_steps == self.n0 + self._new
        elif self.base in ("vol", "orth"):
            ok = lr.n_replay == n_k and lr.n_steps == self.n0 + self._new
        else:
            ok = lr.n_replay == 0 and lr.n_steps == self.n_rounds_billed
        if not ok:
            raise BillingError(f"arm {self.arm}: n_steps={lr.n_steps} n_replay={lr.n_replay} n0={self.n0} "
                               f"new={self._new} N_k={n_k} billed={self.n_rounds_billed}")
        # replay rows are never billed: every billed serial is a real row, no replay serial is in the real ledger
        billed = {o.serial for o, _ in self.real_ledger}
        for r in lr.rows:
            if (r.provenance in REPLAY_PROVS) == (r.serial in billed):
                raise BillingError(f"row {r.serial} provenance {r.provenance} inconsistent with billing")

    def end_problem(self, env_n_steps: int) -> ProblemAccount:
        self.check_billing(env_n_steps)
        acc = self.accounts[-1]
        lr = self.active
        acc.new_env_steps = self._new
        acc.n_rounds_billed = self.n_rounds_billed
        if lr is not None:
            pc = lr.provenance_counts()
            acc.prov_counts = pc
            acc.replay_steps = lr.n_replay
            acc.replay_pad_steps = pc.get(PROV_VOL_PAD, 0)
            acc.n_rounds_total = lr.n_rounds_total
        assert self.cost_by_problem.get(self._pid, 0) == self._new
        self._k = self._pid = None
        self._new = 0
        return acc


def score_truncations(steps: int, certified: bool, taus=TAUS) -> dict:
    """Truncated per-problem cost min(T, tau); a problem not certified within tau new steps is censored at tau."""
    out = {}
    for tau in taus:
        ok = bool(certified and steps <= tau)
        out[f"cost_tau{tau}"] = int(steps if ok else tau)
        out[f"censored_tau{tau}"] = not ok
    return out
