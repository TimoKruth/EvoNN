"""Archive construction for elite, Pareto, and niche archives."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class IndividualSummary:
    """Lightweight summary of an evaluated genome for archive construction."""

    genome_id: str
    family: str
    generation: int
    qualities: dict[str, float]  # benchmark_id -> quality
    parameter_count: int
    train_seconds: float

    @property
    def aggregate_quality(self) -> float:
        if not self.qualities:
            return float("-inf")
        return sum(self.qualities.values()) / len(self.qualities)

    @property
    def best_quality(self) -> float:
        return max(self.qualities.values(), default=float("-inf"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_archives(
    summaries: list[IndividualSummary],
    elite_per_benchmark: int = 3,
) -> dict:
    """Build all archive types from individual summaries.

    Returns dict with keys: "elite", "pareto", "niche".
    """
    scored = _scored(summaries)
    return {
        "elite": build_elite_archive(scored, elite_per_benchmark),
        "pareto": build_pareto_archive(scored),
        "niche": build_niche_archive(scored),
    }


def build_elite_archive(
    summaries: list[IndividualSummary],
    elite_per_benchmark: int,
) -> dict[str, list[IndividualSummary]]:
    """Top K individuals per benchmark by quality."""
    summaries = _scored(_dedupe(summaries))
    grouped: dict[str, list[IndividualSummary]] = defaultdict(list)
    for ind in summaries:
        for benchmark_id in ind.qualities:
            grouped[benchmark_id].append(ind)

    for benchmark_id in grouped:
        grouped[benchmark_id] = sorted(
            grouped[benchmark_id],
            key=lambda s: (s.qualities[benchmark_id], -s.parameter_count),
            reverse=True,
        )[:elite_per_benchmark]

    return dict(grouped)


def build_pareto_archive(
    summaries: list[IndividualSummary],
) -> list[IndividualSummary]:
    """Non-dominated front on (aggregate_quality, -parameter_count).

    An individual is dominated if another individual is at least as good on
    both objectives and strictly better on at least one.
    """
    summaries = _scored(_dedupe(summaries))
    front: list[IndividualSummary] = []

    for candidate in summaries:
        dominated = False
        for other in summaries:
            if other.genome_id == candidate.genome_id:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(candidate)

    front.sort(key=lambda s: s.aggregate_quality, reverse=True)
    return front


def build_niche_archive(
    summaries: list[IndividualSummary],
) -> dict[str, IndividualSummary]:
    """Best individual per family (MAP-Elites style).

    Each family acts as a niche; the individual with the highest aggregate
    quality represents that niche.
    """
    summaries = _scored(_dedupe(summaries))
    archive: dict[str, IndividualSummary] = {}

    for ind in summaries:
        current = archive.get(ind.family)
        if current is None or ind.aggregate_quality > current.aggregate_quality:
            archive[ind.family] = ind

    return archive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dominates(left: IndividualSummary, right: IndividualSummary) -> bool:
    """True if left dominates right on (aggregate_quality, -param_count)."""
    l_vec = (left.aggregate_quality, -left.parameter_count)
    r_vec = (right.aggregate_quality, -right.parameter_count)
    return (
        all(lv >= rv for lv, rv in zip(l_vec, r_vec))
        and any(lv > rv for lv, rv in zip(l_vec, r_vec))
    )


def _dedupe(summaries: list[IndividualSummary]) -> list[IndividualSummary]:
    """Keep the best summary per genome_id."""
    unique: dict[str, IndividualSummary] = {}
    for ind in summaries:
        current = unique.get(ind.genome_id)
        if current is None or ind.aggregate_quality > current.aggregate_quality:
            unique[ind.genome_id] = ind
    return list(unique.values())


def _scored(summaries: list[IndividualSummary]) -> list[IndividualSummary]:
    """Filter to individuals that have at least one quality score."""
    return [s for s in summaries if s.qualities]


def summaries_from_state_results(
    population,
    results: dict,
    generation: int,
) -> list[IndividualSummary]:
    """Build IndividualSummary list from population + results dict.

    Args:
        population: list[ModelGenome]
        results: dict[genome_id -> dict[benchmark_id -> EvaluationResult]]
        generation: Current generation number.
    """
    summaries: list[IndividualSummary] = []
    for genome in population:
        genome_results = results.get(genome.genome_id, {})
        qualities = {
            bid: r.quality
            for bid, r in genome_results.items()
            if r.failure_reason is None
        }
        param_count = max(
            (r.parameter_count for r in genome_results.values()),
            default=0,
        )
        train_secs = sum(r.train_seconds for r in genome_results.values())
        summaries.append(IndividualSummary(
            genome_id=genome.genome_id,
            family=genome.family,
            generation=generation,
            qualities=qualities,
            parameter_count=param_count,
            train_seconds=train_secs,
        ))
    return summaries
