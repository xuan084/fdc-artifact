"""Tests for the guarded FDC-BF entry point (dsswm/baselines/fdc_bet_guarded.py; paper r6c, usage contract).

The legacy ``FDCBet.certify`` (lock-bound, unchanged) computes fresh widths at any non-decreasing time; the guarded
wrapper must (i) equal it on the predeclared checkpoint grid, (ii) refuse off-grid and repeated times, and (iii) leave
dense monitoring to ``FDCTimeUniform``, which accepts off-grid times with widths frozen at the last block point."""
from __future__ import annotations

import numpy as np
import pytest

from dsswm.baselines.fdc_bet import FDCBet, make_variant
from dsswm.baselines.fdc_bet_guarded import (DENSE_MONITORING_CLASS, FDCBetGuarded, OffGridCertifyError,
                                             make_guarded_variant)
from dsswm.baselines.fdc_loc import FDCTimeUniform, dense_grid
from dsswm.baselines.frontier_common import FrontierState
from dsswm.streams.frontier_runner import build_ctx
from dsswm.streams.frontier_runner_v6 import run_stream_v6
from dsswm.tests.test_baselines_r4a import PROBS, TOY_EPS, _toy_population


@pytest.fixture(scope="module")
def toy():
    env = _toy_population(905)
    return env, build_ctx(env, PROBS, TOY_EPS)


def _state(env, ctx, t, frac=0.3):
    n = np.maximum(1, (np.asarray(env.pool_sizes) * frac).astype(np.int64))
    return FrontierState(t, n, n * 0.4, env.pool_sizes, np.ones(ctx.Q, bool), None)


def test_guarded_variant_has_frozen_configuration():
    g, b = make_guarded_variant("FDC-BF"), make_variant("FDC-BF")
    assert isinstance(g, FDCBetGuarded) and isinstance(g, FDCBet)
    assert (g.kind, g.var_box, g.fpc, g.rect, g.split, g.name) == (b.kind, b.var_box, b.fpc, b.rect, b.split, b.name)


def test_guarded_equals_legacy_on_checkpoint_grid(toy):
    env, ctx = toy
    for seed in (905000, 905001):
        s1, r1 = run_stream_v6(env, make_variant("FDC-BF"), seed, PROBS, TOY_EPS, ctx=ctx, keep_U=True)
        s2, r2 = run_stream_v6(env, make_guarded_variant("FDC-BF"), seed, PROBS, TOY_EPS, ctx=ctx, keep_U=True)
        assert s1["N80"] == s2["N80"] and s1["cert_k"] == s2["cert_k"]
        for a, b in zip(r1, r2):
            assert np.array_equal(np.asarray(a["U"]), np.asarray(b["U"]))


def test_guarded_rejects_off_grid_and_repeated_times(toy):
    env, ctx = toy
    ck = [int(x) for x in ctx.checkpoints]
    legacy = make_variant("FDC-BF")
    legacy.setup(ctx)
    legacy.certify(ctx, _state(env, ctx, ck[0]))
    legacy.certify(ctx, _state(env, ctx, ck[0] + 1))              # the legacy API silently issues fresh widths here
    m = make_guarded_variant("FDC-BF")
    m.setup(ctx)
    m.certify(ctx, _state(env, ctx, ck[0]))
    with pytest.raises(OffGridCertifyError, match="FDCTimeUniform"):
        m.certify(ctx, _state(env, ctx, ck[0] + 1))                # off the predeclared grid
    with pytest.raises(OffGridCertifyError):
        m.certify(ctx, _state(env, ctx, ck[0]))                    # not strictly increasing
    out = m.certify(ctx, _state(env, ctx, ck[1]))
    assert len(out) == ctx.Q
    assert DENSE_MONITORING_CLASS.endswith("FDCTimeUniform")


def test_dense_monitoring_goes_through_time_uniform_wrapper(toy):
    env, ctx = toy
    ck = np.asarray(ctx.checkpoints, dtype=np.int64)
    tu = FDCTimeUniform(block_points=ck)
    dense = build_ctx(env, PROBS, TOY_EPS)
    dense.checkpoints = dense_grid(ck, sub=4)
    tu.setup(dense)
    assert tu.ledger["K"] == len(ck)                               # ledger counts block points, not evaluation times
    off = [int(t) for t in dense.checkpoints if int(t) not in set(ck.tolist())]
    assert off
    tu.certify(dense, _state(env, dense, int(ck[0])))
    tu.certify(dense, _state(env, dense, off[0]))                  # accepted: width frozen at the last block point
    assert tu.n_stale_evals == 1
