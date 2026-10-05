"""Evidence boundary: only observations issued by a ground-truth env.step() are authentic.

Model rollouts / simulators can construct Observation objects, but they are never registered,
so evidence sets reject them (EvidenceBoundaryError). The registry stores a digest of the payload,
so a registered observation cannot be mutated and re-used either.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

import numpy as np


class EvidenceBoundaryError(RuntimeError):
    pass


_REGISTRY: dict[int, str] = {}
_COUNTER = itertools.count(1)


@dataclass(frozen=True)
class Observation:
    """One platform round. outcomes: tuple of (i, j, level, y) for matched pairs;
    loads/engaged are BEFORE the action, next_engaged AFTER (engagement envs only)."""
    env_kind: str
    t: int
    loads: tuple
    engaged: tuple
    action: int
    outcomes: tuple
    next_loads: tuple
    next_engaged: tuple
    serial: int = field(default=0)

    def payload(self) -> str:
        d = {k: getattr(self, k) for k in ("env_kind", "t", "loads", "engaged", "action", "outcomes",
                                            "next_loads", "next_engaged", "serial")}
        return json.dumps(d, sort_keys=True, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def _digest(obs: Observation) -> str:
    return hashlib.sha256(obs.payload().encode()).hexdigest()


def issue(**kwargs) -> Observation:
    """Called ONLY by dsswm.envs (ground truth). Returns a registered observation."""
    obs = Observation(serial=next(_COUNTER), **kwargs)
    _REGISTRY[obs.serial] = _digest(obs)
    return obs


def is_authentic(obs) -> bool:
    return isinstance(obs, Observation) and obs.serial in _REGISTRY and _REGISTRY[obs.serial] == _digest(obs)


def require_authentic(obs) -> None:
    if not is_authentic(obs):
        raise EvidenceBoundaryError("observation was not issued by env.step(); model rollouts cannot enter Theta_t")
