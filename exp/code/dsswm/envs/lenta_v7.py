"""Lock-v7 addendum, block B: the LR9 Lenta layer with a v7 eval gate (new file; ``lenta_v6`` is untouched).

``LentaLayerEnvV7`` is ``lenta_v6.LentaLayerEnv`` (same table, sha check, segmentation, split seed 6006, problems,
checkpoints) with one change: eval-half outcomes are released only when ``eval_task_id`` passes
``dsswm.stats.prereg_v7.addendum_gate`` and names a block-B v7 eval task (prefix ``v7b_``).  The object is first
built structure-only by the v6 constructor (no eval outcome materialised), then, after the v7 gate, the eval rows'
outcomes are attached.  Without an authorised task id the eval half stays structure-only (TruthAccessError on every
truth accessor), exactly as in v6.  The dev half behaves exactly like ``LentaLayerEnv('dev')``.
"""
from __future__ import annotations

from .base import TruthAccessError
from .lenta_v6 import LR9_OUTCOME, LentaLayerEnv, _load_lenta

__all__ = ["LentaLayerEnvV7"]


class LentaLayerEnvV7(LentaLayerEnv):
    def __init__(self, half="dev", eval_task_id=None, gate_path=None, check_sha=True):  # noqa: D401
        authorised = False
        if half == "eval" and eval_task_id is not None:
            from ..stats.prereg_v7 import addendum_gate
            ok, info = addendum_gate(eval_task_id, path=gate_path)
            if not ok:
                raise TruthAccessError(f"LR9 eval outcomes refused by the v7 addendum gate: {info}")
            if not str(eval_task_id).startswith("v7b_"):
                raise TruthAccessError("LR9 eval outcomes are reserved for v7 block-B eval tasks")
            authorised = True
        super().__init__(half, eval_task_id=None, gate_path=None, check_sha=check_sha)
        if authorised:
            _, y_full = _load_lenta(check_sha, with_outcome=True)
            v = y_full[self.row_index].copy()
            v.setflags(write=False)
            del y_full
            self._y = {LR9_OUTCOME: v}
            self.structure_only = False
