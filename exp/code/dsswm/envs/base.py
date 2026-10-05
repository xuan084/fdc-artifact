"""Base class + learner-facing handle that hides ground-truth parameters."""
from __future__ import annotations

import numpy as np

from ..core.actions import ActionSpace
from ..core.state import StateCodec


class TruthAccessError(AttributeError):
    pass


class BaseEnv:
    kind = "base"

    def __init__(self, L: int, R: int, nmax: int, engagement: bool, aspace: ActionSpace, seed: int, static: bool = False):
        self.L, self.R, self.P, self.nmax = L, R, L + R, nmax
        self.aspace = aspace
        self.codec = StateCodec(L, R, nmax, engagement)
        self.static = static
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.loads = np.zeros(self.P, dtype=np.int64)
        self.engaged = np.ones(self.P, dtype=np.int64)
        self.n_steps = 0      # real platform rounds consumed
        self.n_resets = 0     # evaluation resets (whole-trial baselines); accounted separately

    # -- public observable state
    def observable_state(self):
        return self.loads.copy(), self.engaged.copy()

    def reset_to(self, loads, engaged=None):
        """Put the platform into a given observable state (used by whole-trial designs).
        Counted in n_resets; steering cost accounting is the caller's responsibility."""
        self.loads = np.asarray(loads, dtype=np.int64).copy()
        self.engaged = (np.ones(self.P, dtype=np.int64) if engaged is None else np.asarray(engaged, dtype=np.int64).copy())
        self.n_resets += 1

    def step(self, a_idx: int):
        raise NotImplementedError

    def handle(self) -> "EnvHandle":
        return EnvHandle(self)


class EnvHandle:
    """Learner-facing proxy. Only step / observable state / public structure are exposed."""
    _PUBLIC = ("step", "observable_state", "reset_to", "aspace", "codec", "L", "R", "P", "nmax", "kind",
               "n_steps", "n_resets", "static", "known_constants")

    def __init__(self, env: BaseEnv):
        object.__setattr__(self, "_EnvHandle__env", env)

    def __getattribute__(self, name):
        if name in EnvHandle._PUBLIC:
            return getattr(object.__getattribute__(self, "_EnvHandle__env"), name)
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        raise TruthAccessError(f"learner may not access env attribute '{name}'")

    def __setattr__(self, name, value):
        raise TruthAccessError("learner may not mutate the environment")
