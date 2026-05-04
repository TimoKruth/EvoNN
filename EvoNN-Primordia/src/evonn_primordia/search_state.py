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
        self.records.sort(key=self._record_priority, reverse=True)

    def elites(self, total_budget: int) -> list[dict[str, Any]]:
        keep = max(1, int(round(max(1, total_budget) * self.elite_fraction)))
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(record: dict[str, Any]) -> None:
            genome_id = str(record.get("genome_id") or "")
            if genome_id and genome_id in selected_ids:
                return
            selected.append(record)
            if genome_id:
                selected_ids.add(genome_id)

        for record in sorted(self.family_best_records().values(), key=self._record_priority, reverse=True):
            if len(selected) >= keep:
                break
            add(record)
        for record in self.records:
            if len(selected) >= keep:
                break
            add(record)
        return selected[:keep]

    def benchmark_best_record(self) -> dict[str, Any] | None:
        return self.records[0] if self.records else None

    def family_best_records(self) -> dict[str, dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for record in self.records:
            family = str(record.get("primitive_family") or "unknown")
            incumbent = best.get(family)
            if incumbent is None or self._record_priority(record) > self._record_priority(incumbent):
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
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(record: dict[str, Any], *, allow_duplicate: bool = False) -> bool:
            genome_id = str(record.get("genome_id") or "")
            if genome_id and genome_id in selected_ids and not allow_duplicate:
                return False
            selected.append(record)
            if genome_id:
                selected_ids.add(genome_id)
            return True

        benchmark_best = self.benchmark_best_record()
        if benchmark_best is not None:
            add(benchmark_best)

        family_best = sorted(self.family_best_records().values(), key=self._record_priority, reverse=True)
        for _ in range(max(0, int(family_exploration_floor))):
            for record in family_best:
                if len(selected) >= count:
                    break
                add(record)
            if len(selected) >= count:
                break

        weighted_pool = sorted(elites, key=self._record_priority, reverse=True)
        while len(selected) < count:
            candidates = [record for record in weighted_pool if str(record.get("genome_id") or "") not in selected_ids]
            allow_duplicate = False
            if not candidates:
                candidates = weighted_pool
                allow_duplicate = True
            selected_record = self._weighted_choice(candidates, rng)
            add(selected_record, allow_duplicate=allow_duplicate)
        return selected[:count]

    @staticmethod
    def _record_priority(record: dict[str, Any]) -> tuple[float, float, int, float]:
        search_score = float(record.get("search_score", float("-inf")))
        novelty = float(record.get("novelty_score", 0.0) or 0.0)
        generation = int(record.get("generation", 0) or 0)
        complexity = -float(record.get("complexity_penalty", 0.0) or 0.0)
        return (search_score, novelty, generation, complexity)

    @classmethod
    def _selection_weight(cls, record: dict[str, Any]) -> float:
        search_score, novelty, generation, complexity = cls._record_priority(record)
        finite_score = search_score if search_score > float("-inf") else -1e6
        return max(1e-6, finite_score + 1e6) + novelty + max(0, generation) + max(0.0, complexity)

    @classmethod
    def _weighted_choice(cls, records: list[dict[str, Any]], rng: Random) -> dict[str, Any]:
        if len(records) == 1:
            return records[0]
        weights = [cls._selection_weight(record) for record in records]
        total = sum(weights)
        if total <= 0:
            return rng.choice(records)
        needle = rng.random() * total
        running = 0.0
        for record, weight in zip(records, weights, strict=False):
            running += weight
            if needle <= running:
                return record
        return records[-1]
