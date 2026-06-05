"""Selection and reproduction for Prism evolution."""

from __future__ import annotations

from collections import Counter
from math import log1p
from random import Random

from prism.genome import ModelGenome, apply_random_mutation, crossover
from prism.pipeline.archive import IndividualSummary

MAX_OFFSPRING_ATTEMPTS = 12


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
    parent_pool, quality_map = _build_parent_pool(state, archives, evolution)
    quality_map = _apply_selection_pressure(
        state,
        quality_map,
        undercovered_bias=evolution.undercovered_parent_bias,
        family_diversity_bias=evolution.family_diversity_bias,
        family_stale_penalty=evolution.family_stale_penalty,
        novelty_bias=evolution.novelty_parent_bias,
        family_prior_bias=evolution.family_prior_bias,
    )

    if not parent_pool:
        # Fallback: use current population
        parent_pool = list(state.population)
        quality_map = _quality_map_from_results(state, evolution)

    offspring: list[ModelGenome] = []
    lineage: list[dict] = []
    seen_offspring_ids: set[str] = set()
    parent_pool_ids = {genome.genome_id for genome in parent_pool}
    family_floor_targets = _family_floor_targets(parent_pool, quality_map, evolution.family_offspring_floor)
    specialist_targets = _benchmark_specialist_targets(
        state,
        evolution.benchmark_specialist_offspring,
        repair_fraction=evolution.benchmark_specialist_repair_fraction,
        exploit_min_quality=evolution.benchmark_specialist_exploit_min_quality,
        exploit_saturation=evolution.benchmark_specialist_exploit_saturation,
    )

    for slot in range(evolution.offspring_per_generation):
        family_target = family_floor_targets[slot] if slot < len(family_floor_targets) else None
        specialist_target = specialist_targets[slot] if slot < len(specialist_targets) else None
        child, record = _select_novel_offspring(
            parent_pool,
            quality_map,
            evolution,
            rng,
            state=state,
            seen_offspring_ids=seen_offspring_ids,
            parent_pool_ids=parent_pool_ids,
            family_target=family_target,
            specialist_target=specialist_target,
        )
        offspring.append(child)
        lineage.append(record)
        seen_offspring_ids.add(child.genome_id)

    return offspring, lineage


def _select_novel_offspring(
    parent_pool: list[ModelGenome],
    quality_map: dict[str, float],
    evolution,
    rng: Random,
    state,
    seen_offspring_ids: set[str],
    parent_pool_ids: set[str],
    family_target: str | None,
    specialist_target: dict | None,
) -> tuple[ModelGenome, dict]:
    strategies = [
        (family_target, specialist_target),
        (family_target, None),
        (None, None),
    ]
    for target_family, target_specialist in strategies:
        for _attempt in range(MAX_OFFSPRING_ATTEMPTS):
            child, record = _make_offspring(
                parent_pool,
                quality_map,
                evolution,
                rng,
                state=state,
                family_target=target_family,
                specialist_target=target_specialist,
            )
            if not _is_novel_offspring(
                child,
                record.get("parent_ids", []),
                seen_offspring_ids,
                parent_pool_ids,
            ):
                continue
            if target_family is not None and child.family != target_family:
                continue
            return child, record
    raise RuntimeError(
        "failed to generate a novel Prism offspring after exhausting reproduction retries"
    )


def _make_offspring(
    parent_pool: list[ModelGenome],
    quality_map: dict[str, float],
    evolution,
    rng: Random,
    state,
    family_target: str | None = None,
    specialist_target: dict | None = None,
) -> tuple[ModelGenome, dict]:
    if specialist_target is not None:
        benchmark_id = specialist_target["benchmark_id"]
        candidate_ids = specialist_target["genome_ids"]
        specialist_pool = [genome for genome in parent_pool if genome.genome_id in candidate_ids]
        if specialist_pool:
            parent = max(specialist_pool, key=lambda genome: quality_map.get(genome.genome_id, float("-inf")))
            child, op_name = apply_random_mutation(
                parent,
                evolution,
                rng,
                operator_weights=_operator_weights_for_parent(state, parent),
            )
            return child, {
                "genome_id": child.genome_id,
                "parent_ids": [parent.genome_id],
                "operator": f"specialist:{benchmark_id}:mutation:{op_name}",
            }

    if family_target is not None:
        parent = _best_in_family(parent_pool, quality_map, family_target)
        child, op_name = apply_random_mutation(
            parent,
            evolution,
            rng,
            operator_weights=_operator_weights_for_parent(state, parent),
        )
        return child, {
            "genome_id": child.genome_id,
            "parent_ids": [parent.genome_id],
            "operator": f"mutation:{op_name}",
        }

    if rng.random() < evolution.crossover_rate and len(parent_pool) >= 2:
        p1 = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
        p2 = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
        attempts = 0
        while p2.genome_id == p1.genome_id and attempts < 5:
            p2 = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
            attempts += 1

        child = crossover(p1, p2, rng)
        return child, {
            "genome_id": child.genome_id,
            "parent_ids": [p1.genome_id, p2.genome_id],
            "operator": "crossover",
        }

    parent = tournament_select(parent_pool, quality_map, evolution.tournament_size, rng)
    child, op_name = apply_random_mutation(
        parent,
        evolution,
        rng,
        operator_weights=_operator_weights_for_parent(state, parent),
    )
    return child, {
        "genome_id": child.genome_id,
        "parent_ids": [parent.genome_id],
        "operator": f"mutation:{op_name}",
    }


def _is_novel_offspring(
    child: ModelGenome,
    parent_ids: list[str],
    seen_offspring_ids: set[str],
    parent_pool_ids: set[str],
) -> bool:
    if child.genome_id in seen_offspring_ids:
        return False
    if child.genome_id in parent_pool_ids:
        return False
    return child.genome_id not in set(parent_ids)


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
    evolution,
) -> tuple[list[ModelGenome], dict[str, float]]:
    """Build a diverse parent pool from archives and population.

    Pool includes: Pareto front members, per-benchmark elites,
    niche representatives, and the current population.
    """
    genome_map: dict[str, ModelGenome] = {g.genome_id: g for g in state.population}
    quality_map: dict[str, float] = _quality_map_from_results(state, evolution)

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
        _add(summary.genome_id, summary.search_quality)

    # Per-benchmark elites
    elite_archive: dict[str, list[IndividualSummary]] = archives.get("elite", {})
    for elites in elite_archive.values():
        for summary in elites:
            _add(summary.genome_id, summary.search_quality)

    # Niche representatives
    niche_archive: dict[str, IndividualSummary] = archives.get("niche", {})
    for summary in niche_archive.values():
        _add(summary.genome_id, summary.search_quality)

    efficient_archive = archives.get("efficient", {})
    for summary in efficient_archive.get("family", {}).values():
        _add(summary.genome_id, summary.search_quality)
    for summaries in efficient_archive.get("benchmark", {}).values():
        for summary in summaries:
            _add(summary.genome_id, summary.search_quality)

    # Current population (ensures pool is never empty)
    for genome in state.population:
        _add(genome.genome_id)

    return pool, quality_map


def _quality_map_from_results(state, evolution) -> dict[str, float]:
    """Extract efficiency-adjusted parent score per genome from state.results."""
    profiles: dict[str, dict[str, float]] = {}
    genomes_by_id = {genome.genome_id: genome for genome in getattr(state, "population", [])}
    normalized_quality = _normalized_genome_quality_scores(state.results)
    for genome_id, benchmark_results in state.results.items():
        valid = [result for result in benchmark_results.values() if result.failure_reason is None]
        if not valid:
            continue
        genome = genomes_by_id.get(genome_id)
        avg_quality = normalized_quality.get(genome_id)
        if avg_quality is None:
            avg_quality = sum(result.quality for result in valid) / len(valid)
        avg_time = sum(result.train_seconds for result in valid) / len(valid)
        avg_params = sum(result.parameter_count for result in valid) / len(valid)
        profiles[genome_id] = {
            "quality": avg_quality,
            "time": avg_time,
            "params": avg_params,
            "complexity": float(getattr(genome, "architecture_complexity", 0.0)),
        }

    if not profiles:
        return {}

    bias = _efficiency_bias(
        getattr(state, "generation", 0),
        getattr(evolution, "num_generations", 1),
        evolution.efficiency_bias_start,
        evolution.efficiency_bias_end,
        evolution.efficiency_warmup_generations,
    )
    time_logs = [log1p(profile["time"]) for profile in profiles.values()]
    param_logs = [log1p(profile["params"]) for profile in profiles.values()]
    complexity_values = [profile["complexity"] for profile in profiles.values()]
    time_weight = evolution.time_penalty_weight
    param_weight = evolution.param_penalty_weight
    complexity_weight = evolution.complexity_penalty_weight
    total_weight = max(1e-9, time_weight + param_weight + complexity_weight)

    quality_map: dict[str, float] = {}
    for genome_id, profile in profiles.items():
        time_penalty = _normalized_range(log1p(profile["time"]), time_logs)
        param_penalty = _normalized_range(log1p(profile["params"]), param_logs)
        complexity_penalty = _normalized_range(profile["complexity"], complexity_values)
        efficiency_penalty = (
            (time_weight * time_penalty)
            + (param_weight * param_penalty)
            + (complexity_weight * complexity_penalty)
        ) / total_weight
        quality_map[genome_id] = profile["quality"] - (bias * efficiency_penalty)
    return quality_map


def _normalized_genome_quality_scores(results: dict) -> dict[str, float]:
    by_benchmark: dict[str, list[tuple[str, float]]] = {}
    for genome_id, benchmark_results in results.items():
        for benchmark_id, result in benchmark_results.items():
            if result.failure_reason is None:
                by_benchmark.setdefault(benchmark_id, []).append((genome_id, float(result.quality)))

    per_genome: dict[str, list[float]] = {}
    for values in by_benchmark.values():
        qualities = [quality for _, quality in values]
        lo = min(qualities)
        hi = max(qualities)
        for genome_id, quality in values:
            score = _quality_score(quality, lo, hi)
            per_genome.setdefault(genome_id, []).append(score)

    return {
        genome_id: sum(scores) / len(scores)
        for genome_id, scores in per_genome.items()
        if scores
    }


def _quality_score(quality: float, lo: float, hi: float) -> float:
    if 0.0 <= lo <= hi <= 1.0:
        return quality
    return 1.0 if hi <= lo + 1e-12 else (quality - lo) / (hi - lo)


def _apply_selection_pressure(
    state,
    quality_map: dict[str, float],
    *,
    undercovered_bias: float,
    family_diversity_bias: float,
    family_stale_penalty: float,
    novelty_bias: float,
    family_prior_bias: float,
) -> dict[str, float]:
    """Adjust parent scores toward undercovered, rare-family, and novel genomes."""
    if not state.results:
        return quality_map

    boosted = dict(quality_map)
    if not boosted:
        return boosted

    genomes_by_id = {genome.genome_id: genome for genome in state.population}
    success_counts: dict[str, int] = {}
    for benchmark_results in state.results.values():
        for benchmark_id, result in benchmark_results.items():
            if result.failure_reason is None:
                success_counts[benchmark_id] = success_counts.get(benchmark_id, 0) + 1

    population_size = max(1, len(state.population))
    scarcity = {
        benchmark_id: 1.0 - min(1.0, count / population_size)
        for benchmark_id, count in success_counts.items()
    }
    family_counts = Counter(genome.family for genome in state.population)
    layer_patterns = Counter(tuple(genome.hidden_layers) for genome in state.population)

    for genome_id, benchmark_results in state.results.items():
        if genome_id not in boosted:
            continue
        genome = genomes_by_id.get(genome_id)
        if genome is None:
            continue
        benchmark_scarcity = [
            scarcity[benchmark_id]
            for benchmark_id, result in benchmark_results.items()
            if result.failure_reason is None and benchmark_id in scarcity
        ]
        if benchmark_scarcity:
            boosted[genome_id] += undercovered_bias * (
                sum(benchmark_scarcity) / len(benchmark_scarcity)
            )

        family_ratio = family_counts[genome.family] / population_size
        boosted[genome_id] += family_diversity_bias * (1.0 - family_ratio)
        if family_counts[genome.family] > 1:
            boosted[genome_id] -= family_stale_penalty * (family_counts[genome.family] - 1) / population_size

        family_stats = getattr(state, "family_stats", {}).get(genome.family, {})
        family_count = family_stats.get("count", 0.0)
        if family_count > 0:
            family_avg = family_stats.get("efficiency_sum", family_stats.get("quality_sum", 0.0)) / family_count
            family_fail = family_stats.get("failures", 0.0) / family_count
            boosted[genome_id] += family_prior_bias * family_avg
            boosted[genome_id] -= family_prior_bias * 0.5 * family_fail

        pattern_ratio = layer_patterns[tuple(genome.hidden_layers)] / population_size
        boosted[genome_id] += novelty_bias * (1.0 - pattern_ratio)

    return boosted


def _best_in_family(
    pool: list[ModelGenome],
    quality_map: dict[str, float],
    family: str,
) -> ModelGenome:
    members = [genome for genome in pool if genome.family == family]
    if not members:
        raise ValueError(f"Family {family!r} not present in parent pool")
    return max(members, key=lambda genome: quality_map.get(genome.genome_id, float("-inf")))


def _family_floor_targets(
    pool: list[ModelGenome],
    quality_map: dict[str, float],
    per_family_floor: int,
) -> list[str]:
    if per_family_floor <= 0:
        return []

    by_family: dict[str, list[ModelGenome]] = {}
    for genome in pool:
        by_family.setdefault(genome.family, []).append(genome)

    families = sorted(
        by_family,
        key=lambda family: max(
            quality_map.get(genome.genome_id, float("-inf"))
            for genome in by_family[family]
        ),
        reverse=True,
    )
    return [family for family in families for _ in range(per_family_floor)]


def _benchmark_specialist_targets(
    state,
    specialist_slots: int,
    *,
    repair_fraction: float = 0.5,
    exploit_min_quality: float = 0.75,
    exploit_saturation: float = 0.995,
) -> list[dict]:
    if specialist_slots <= 0:
        return []

    specialist_archive: dict[str, dict[str, IndividualSummary]] = getattr(state, "archives", {}).get("specialist", {})
    if specialist_archive:
        benchmark_scores = {
            benchmark_id: [
                (summary.genome_id, summary.qualities.get(benchmark_id, float("-inf")))
                for summary in summaries.values()
            ]
            for benchmark_id, summaries in specialist_archive.items()
        }
        return _rank_benchmark_specialist_targets(
            benchmark_scores,
            specialist_slots,
            repair_fraction=repair_fraction,
            exploit_min_quality=exploit_min_quality,
            exploit_saturation=exploit_saturation,
        )

    benchmark_scores: dict[str, list[tuple[str, float]]] = {}
    for genome_id, benchmark_results in state.results.items():
        for benchmark_id, result in benchmark_results.items():
            if result.failure_reason is None:
                benchmark_scores.setdefault(benchmark_id, []).append((genome_id, result.quality))

    if not benchmark_scores:
        return []

    return _rank_benchmark_specialist_targets(
        benchmark_scores,
        specialist_slots,
        repair_fraction=repair_fraction,
        exploit_min_quality=exploit_min_quality,
        exploit_saturation=exploit_saturation,
    )


def _rank_benchmark_specialist_targets(
    benchmark_scores: dict[str, list[tuple[str, float]]],
    specialist_slots: int,
    *,
    repair_fraction: float,
    exploit_min_quality: float,
    exploit_saturation: float,
) -> list[dict]:
    if not benchmark_scores or specialist_slots <= 0:
        return []

    repair_slots = min(
        specialist_slots,
        max(1, int(round(specialist_slots * max(0.0, min(1.0, repair_fraction))))),
    )
    exploit_slots = max(0, specialist_slots - repair_slots)

    repair_ranked = sorted(
        benchmark_scores,
        key=lambda benchmark_id: _average_score(benchmark_scores[benchmark_id]),
    )

    targets: list[dict] = []
    selected: set[str] = set()

    for benchmark_id in repair_ranked[:repair_slots]:
        targets.append(_specialist_target_payload(benchmark_id, benchmark_scores[benchmark_id]))
        selected.add(benchmark_id)

    if exploit_slots:
        exploit_ranked = sorted(
            (
                (benchmark_id, _exploitability_score(scores, exploit_min_quality, exploit_saturation))
                for benchmark_id, scores in benchmark_scores.items()
                if benchmark_id not in selected
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for benchmark_id, score in exploit_ranked:
            if score <= float("-inf"):
                continue
            targets.append(_specialist_target_payload(benchmark_id, benchmark_scores[benchmark_id]))
            selected.add(benchmark_id)
            if len(targets) >= specialist_slots:
                break

    if len(targets) < specialist_slots:
        for benchmark_id in repair_ranked:
            if benchmark_id in selected:
                continue
            targets.append(_specialist_target_payload(benchmark_id, benchmark_scores[benchmark_id]))
            if len(targets) >= specialist_slots:
                break
    return targets


def _specialist_target_payload(benchmark_id: str, scores: list[tuple[str, float]]) -> dict:
    specialists = sorted(scores, key=lambda item: item[1], reverse=True)
    return {
        "benchmark_id": benchmark_id,
        "genome_ids": [genome_id for genome_id, _ in specialists[:2]],
    }


def _average_score(scores: list[tuple[str, float]]) -> float:
    if not scores:
        return float("-inf")
    return sum(score for _, score in scores) / len(scores)


def _exploitability_score(
    scores: list[tuple[str, float]],
    exploit_min_quality: float,
    exploit_saturation: float,
) -> float:
    values = [score for _, score in scores]
    if not values:
        return float("-inf")

    best = max(values)
    if not 0.0 <= min(values) <= best <= 1.0:
        return float("-inf")
    if best < exploit_min_quality or best >= exploit_saturation:
        return float("-inf")

    avg = sum(values) / len(values)
    spread = best - min(values)
    headroom = max(0.0, exploit_saturation - best)
    return best + (0.25 * spread) + (0.1 * headroom) - (0.15 * avg)


def _operator_weights_for_parent(state, parent: ModelGenome) -> dict[str, float]:
    weights: dict[str, float] = {}
    operator_stats = getattr(state, "operator_stats", {})
    family_stats = getattr(state, "family_stats", {}).get(parent.family, {})
    family_count = max(1.0, family_stats.get("count", 0.0))
    family_failure_rate = family_stats.get("failures", 0.0) / family_count
    family_quality = family_stats.get("efficiency_sum", family_stats.get("quality_sum", 0.0)) / family_count if family_count else 0.0

    for operator, payload in operator_stats.items():
        bucket_count = max(1.0, payload.get("count", 0.0))
        operator_quality = payload.get("efficiency_sum", payload.get("quality_sum", 0.0)) / bucket_count if bucket_count else 0.0
        operator_failure = payload.get("failures", 0.0) / bucket_count
        label = operator.rsplit(":", 1)[-1] if ":" in operator else operator
        weights[label] = max(0.05, 1.0 + operator_quality - (operator_failure * 0.5))

    if family_quality > 0:
        for label in list(weights):
            weights[label] += family_quality * 0.1
    if family_failure_rate > 0:
        for label in list(weights):
            weights[label] = max(0.05, weights[label] - family_failure_rate * 0.1)
    return weights


def _efficiency_bias(
    generation: int,
    total_generations: int,
    start: float,
    end: float,
    warmup_generations: int,
) -> float:
    if total_generations <= 1:
        return end
    if generation < warmup_generations:
        return start
    active_total = max(1, total_generations - 1 - warmup_generations)
    progress = min(1.0, max(0.0, (generation - warmup_generations) / active_total))
    return start + ((end - start) * progress)


def _normalized_range(value: float, values: list[float]) -> float:
    if not values:
        return 0.5
    lo = min(values)
    hi = max(values)
    if hi <= lo + 1e-9:
        return 1.0
    return (value - lo) / (hi - lo)
