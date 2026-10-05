"""Replay check before a zero-cost certificate from ledger reuse (methodology section 4.2).

Before a new problem is certified at zero cost from the reused ledger, the mixed set is recomputed on the FULL
ledger (all real rounds so far, in order) with a regime map that includes the current problem id, and the check
fails if the recomputed set is empty. No new interaction happens: when an env handle is given, its step counter is
asserted unchanged. The recomputation re-validates every round's provenance (require_authentic).
"""
from __future__ import annotations

from .mixed_lr import COMPONENTS, MixedLRSet
from .regimes import load_bucket


def replay_check(prop, LT, ledger, current_pid, delta, components=COMPONENTS, regime_fn=load_bucket,
                 env_handle=None, **mixed_kwargs) -> dict:
    """ledger: iterable of (obs, problem_id) in collection order (problem_id None for shared initial data).
    Returns {'conflict': bool, 'size': int, 'mask': tensor, 'n_replayed': int, 'set': MixedLRSet}."""
    n0 = env_handle.n_steps if env_handle is not None else None
    ms = MixedLRSet(prop, LT, delta, components=components, regime_fn=regime_fn, **mixed_kwargs)
    n = 0
    for obs, pid in ledger:
        ms.update(obs, problem_id=pid)
        n += 1
    ms.replay_current_pid = current_pid
    if env_handle is not None:
        assert env_handle.n_steps == n0, "replay check must not interact with the platform"
    size = ms.size()
    return {"conflict": size == 0, "size": size, "mask": ms.mask(), "n_replayed": n, "set": ms,
            "current_pid": current_pid, "scope": ms.falsification_scope()}
