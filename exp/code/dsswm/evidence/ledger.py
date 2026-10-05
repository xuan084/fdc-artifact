"""Evidence ledger: the single entry point through which real interactions reach evidence sets,
with per-problem cost accounting (new env steps), tags (initial / probe / trial / audit) and history."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LedgerEntry:
    serial: int
    problem_id: str | None
    tag: str
    t: int
    cost: float


@dataclass
class EvidenceLedger:
    evidence_sets: list
    entries: list = field(default_factory=list)
    cost_by_problem: dict = field(default_factory=lambda: defaultdict(float))
    steer_cost_by_problem: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, obs, tag: str, problem_id: str | None = None, cost: float = 1.0) -> None:
        for ev in self.evidence_sets:
            ev.update(obs)     # each set enforces the provenance check
        self.entries.append(LedgerEntry(obs.serial, problem_id, tag, obs.t, cost))
        if problem_id is not None:
            self.cost_by_problem[problem_id] += cost

    def record_steer(self, problem_id: str, cost: float) -> None:
        self.steer_cost_by_problem[problem_id] += cost

    def total_new_steps(self, exclude_tags=("initial",)) -> float:
        return sum(e.cost for e in self.entries if e.tag not in exclude_tags)

    def summary(self) -> dict:
        tags = defaultdict(int)
        for e in self.entries:
            tags[e.tag] += 1
        return {"n_entries": len(self.entries), "by_tag": dict(tags),
                "cost_by_problem": dict(self.cost_by_problem), "steer_cost_by_problem": dict(self.steer_cost_by_problem)}
