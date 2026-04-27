"""Search-state helpers for Primordia benchmark-local evolution."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any


@dataclass(frozen=True)
class CandidateSeed:
    genome: Any
    generation: int
    parent_genome_id: str | None
    mutation_operator: str | None


class EliteArchive:
    def __init__(self, elite_fraction: float) -> None:
        self.elite_fraction = max(0.1, min(1.0, float(elite_fraction)))
        self.records: list[dict[str, Any]] = []

    def update(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.records.sort(key=lambda item: float(item.get("search_score", float("-inf"))), reverse=True)

    def elites(self, total_budget: int) -> list[dict[str, Any]]:
        keep = max(1, int(round(max(1, total_budget) * self.elite_fraction)))
        return list(self.records[:keep])

    def family_best_records(self) -> dict[str, dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for record in self.records:
            family = str(record.get("primitive_family") or "unknown")
            incumbent = best.get(family)
            if incumbent is None or float(record.get("search_score", float("-inf"))) > float(incumbent.get("search_score", float("-inf"))):
                best[family] = record
        return best

    def sample_parent_records(
        self,
        *,
        count: int,
        total_budget: int,
        rng: Random,
        family_exploration_floor: int,
    ) -> list[dict[str, Any]]:
        elites = self.elites(total_budget)
        if not elites:
            return []
        family_best = self.family_best_records()
        selected: list[dict[str, Any]] = []
        for family in sorted(family_best):
            if len(selected) >= count:
                break
            if family_exploration_floor <= 0:
                break
            selected.append(family_best[family])
            if len([row for row in selected if row.get("primitive_family") == family]) >= family_exploration_floor:
                continue
        weighted_pool = sorted(elites, key=lambda item: float(item.get("search_score", float("-inf"))), reverse=True)
        while len(selected) < count:
            selected.append(rng.choice(weighted_pool[: max(1, min(len(weighted_pool), count))]))
        return selected[:count]
