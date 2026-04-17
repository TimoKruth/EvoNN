"""Selection and reproduction for Prism evolution."""

from __future__ import annotations

from random import Random

from prism.genome import ModelGenome, apply_random_mutation, crossover
from prism.pipeline.archive import IndividualSummary


def reproduce(
    state,
    config,
    rng: Random,
) -> tuple[list[ModelGenome], list[dict]]:
    """Create offspring from current population.

    1. Build parent pool from Pareto front + elites + undercovered elites.
    2. For each offspring slot:
       - With crossover_rate: select 2 parents via tournament, crossover.
       - Otherwise: select 1 parent via tournament, mutate.
    3. Track operator used for each offspring.

    Returns:
        (new_genomes, lineage_records) where each lineage record is a dict
        with keys: genome_id, parent_ids, operator.
    """
    evolution = config.evolution
    archives = state.archives

    # Build parent pool from archives
    parent_pool, quality_map = _build_parent_pool(state, archives)

    if not parent_pool:
        # Fallback: use current population
        parent_pool = list(state.population)
        quality_map = _quality_map_from_results(state)

    offspring: list[ModelGenome] = []
    lineage: list[dict] = []

    for _ in range(evolution.offspring_per_generation):
        if rng.random() < evolution.crossover_rate and len(parent_pool) >= 2:
            # Crossover
            p1 = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
            p2 = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
            # Avoid self-crossover
            attempts = 0
            while p2.genome_id == p1.genome_id and attempts < 5:
                p2 = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
                attempts += 1

            child = crossover(p1, p2, rng)
            lineage.append({
                "genome_id": child.genome_id,
                "parent_ids": [p1.genome_id, p2.genome_id],
                "operator": "crossover",
            })
        else:
            # Mutation
            parent = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
            child, op_name = apply_random_mutation(parent, evolution, rng)
            lineage.append({
                "genome_id": child.genome_id,
                "parent_ids": [parent.genome_id],
                "operator": f"mutation:{op_name}",
            })

        offspring.append(child)

    return offspring, lineage


def tournament_select(
    pool: list[ModelGenome],
    qualities: dict[str, float],
    tournament_size: int,
    rng: Random,
) -> ModelGenome:
    """Tournament selection: pick the best of K random individuals.

    Quality is looked up by genome_id. Individuals without a quality
    score receive -inf (worst possible).
    """
    k = min(tournament_size, len(pool))
    contestants = rng.sample(pool, k)
    return max(contestants, key=lambda g: qualities.get(g.genome_id, float("-inf")))


def _build_parent_pool(
    state,
    archives: dict,
) -> tuple[list[ModelGenome], dict[str, float]]:
    """Build a diverse parent pool from archives and population.

    Pool includes: Pareto front members, per-benchmark elites,
    niche representatives, and the current population.
    """
    genome_map: dict[str, ModelGenome] = {g.genome_id: g for g in state.population}
    quality_map: dict[str, float] = _quality_map_from_results(state)

    pool_ids: set[str] = set()
    pool: list[ModelGenome] = []

    def _add(genome_id: str, quality: float | None = None) -> None:
        if genome_id in pool_ids:
            return
        genome = genome_map.get(genome_id)
        if genome is None:
            return
        pool_ids.add(genome_id)
        pool.append(genome)
        if quality is not None and genome_id not in quality_map:
            quality_map[genome_id] = quality

    # Pareto front
    pareto: list[IndividualSummary] = archives.get("pareto", [])
    for summary in pareto:
        _add(summary.genome_id, summary.aggregate_quality)

    # Per-benchmark elites
    elite_archive: dict[str, list[IndividualSummary]] = archives.get("elite", {})
    for elites in elite_archive.values():
        for summary in elites:
            _add(summary.genome_id, summary.aggregate_quality)

    # Niche representatives
    niche_archive: dict[str, IndividualSummary] = archives.get("niche", {})
    for summary in niche_archive.values():
        _add(summary.genome_id, summary.aggregate_quality)

    # Current population (ensures pool is never empty)
    for genome in state.population:
        _add(genome.genome_id)

    return pool, quality_map


def _quality_map_from_results(state) -> dict[str, float]:
    """Extract aggregate quality per genome from state.results."""
    quality_map: dict[str, float] = {}
    for genome_id, benchmark_results in state.results.items():
        qualities = [
            r.quality for r in benchmark_results.values()
            if r.failure_reason is None
        ]
        if qualities:
            quality_map[genome_id] = sum(qualities) / len(qualities)
    return quality_map
