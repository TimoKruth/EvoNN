"""Main evolution loop: setup, evaluate, archive, reproduce, checkpoint."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from random import Random
from uuid import uuid4

from prism.config import RunConfig
from prism.genome import ModelGenome, apply_random_mutation, create_seed_genome
from prism.families.compiler import compatible_families
from prism.monitor import TerminalMonitor
from prism.pipeline.archive import (
    build_archives,
    summaries_from_state_results,
)
from prism.pipeline.evaluate import GenerationState, evaluate, select_benchmarks
from prism.pipeline.reproduce import reproduce
from prism.runtime.cache import WeightCache
from prism.storage import RunStore


def run_evolution(
    config: RunConfig,
    benchmark_specs: list,
    run_dir: str | None = None,
    resume: bool = False,
) -> GenerationState:
    """Run full evolution. Returns final GenerationState.

    Args:
        config: RunConfig with training, evolution, and benchmark settings.
        benchmark_specs: List of benchmark specifications to evaluate against.
        run_dir: Optional directory for checkpoints and logs.
        resume: If True, attempt to resume from checkpoint in run_dir.

    Returns:
        Final GenerationState with results and archives.
    """
    config = _resolved_run_config(config, benchmark_specs)
    rng = Random(config.seed)
    evolution = config.evolution
    training = config.training

    # Setup run directory
    if run_dir is None:
        run_dir = f"runs/prism-{uuid4().hex[:8]}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    run_id = Path(run_dir).name

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    Path(config_path).write_text(
        json.dumps(config.model_dump(), indent=2),
        encoding="utf-8",
    )

    store = RunStore(Path(run_dir) / "metrics.duckdb")
    store.save_run(run_id, config.model_dump(mode="json"))

    # Weight cache
    cache = WeightCache() if training.weight_inheritance else None

    # Monitor
    monitor = TerminalMonitor()

    # Resume from checkpoint
    start_gen = 0
    state: GenerationState | None = None

    if resume:
        state, start_gen = _try_resume(run_dir)
        if state is not None:
            monitor.on_info(f"Resuming from generation {start_gen}")

    # Create seed population
    if state is None:
        population = _create_seed_population(evolution, rng)
        state = GenerationState(
            generation=0,
            population=population,
            parent_ids={g.genome_id: [] for g in population},
        )

    run_start = time.time()

    try:
        # Main evolution loop
        total_gens = evolution.num_generations
        for gen in range(start_gen, total_gens):
            state.generation = gen

            # 1. Select benchmarks (undercovered focus)
            gen_benchmarks = select_benchmarks(state, config, benchmark_specs, rng)

            # 2. Evaluate (with cache + multi-fidelity)
            state = evaluate(
                state,
                config,
                gen_benchmarks,
                cache,
                store=store,
                run_id=run_id,
            )

            # 3. Build archives (elite + pareto + niche)
            summaries = summaries_from_state_results(
                state.population, state.results, gen,
            )
            state.archives = build_archives(
                summaries,
                elite_per_benchmark=evolution.elite_per_benchmark,
            )
            _persist_archives(store, run_id, state.archives)

            # 4. Monitor
            elapsed = time.time() - run_start
            qualities = [s.aggregate_quality for s in summaries if s.qualities]
            best_q = max(qualities) if qualities else float("-inf")
            avg_q = sum(qualities) / len(qualities) if qualities else float("-inf")
            families = sorted({g.family for g in state.population})

            monitor.on_generation(
                gen=gen,
                total=total_gens,
                best_quality=best_q,
                avg_quality=avg_q,
                families_active=families,
                population_size=len(state.population),
                elapsed=elapsed,
            )

            # 5. Checkpoint
            _checkpoint(run_dir, gen, state)

            # 6. Reproduce (skip on last generation)
            if gen < total_gens - 1:
                offspring, lineage = reproduce(state, config, rng)
                _persist_lineage(store, run_id, gen + 1, lineage)

                # Replace population with offspring
                state.population = offspring
                state.parent_ids = {
                    record["genome_id"]: list(record.get("parent_ids", []))
                    for record in lineage
                }

                # Clear per-genome results for the new population
                active_ids = {g.genome_id for g in state.population}
                state.results = {
                    gid: res for gid, res in state.results.items()
                    if gid in active_ids
                }

        # Completion
        elapsed = time.time() - run_start
        qualities = [
            s.aggregate_quality
            for s in summaries_from_state_results(state.population, state.results, total_gens - 1)
            if s.qualities
        ]
        best_q = max(qualities) if qualities else float("-inf")
        monitor.on_complete(best_q, total_gens, elapsed)

        # Write final summary
        _write_summary(run_dir, state, elapsed)

        return state
    finally:
        store.close()


def _create_seed_population(
    evolution,
    rng: Random,
) -> list[ModelGenome]:
    """Create initial population: one genome per allowed family, then fill randomly."""
    allowed = evolution.allowed_families or ["mlp", "conv2d", "attention"]

    population: list[ModelGenome] = []
    seen_ids: set[str] = set()

    # One seed per family
    for family in allowed:
        genome = create_seed_genome(family, evolution, rng)
        if genome.genome_id in seen_ids:
            continue
        population.append(genome)
        seen_ids.add(genome.genome_id)

    # Fill remaining slots via mutation from diverse seeds
    while len(population) < evolution.population_size:
        parent = rng.choice(population)
        child = None
        for _ in range(32):
            candidate, _ = apply_random_mutation(parent, evolution, rng)
            if candidate.genome_id not in seen_ids:
                child = candidate
                break
            parent = rng.choice(population)
        if child is None:
            break
        population.append(child)
        seen_ids.add(child.genome_id)

    return population[: evolution.population_size]


def _checkpoint(run_dir: str, generation: int, state: GenerationState) -> None:
    """Save generation state to checkpoint file."""
    checkpoint = {
        "generation": generation,
        "population": [g.model_dump() for g in state.population],
        "parent_ids": state.parent_ids,
        "results": {
            genome_id: {
                bid: {
                    "metric_name": r.metric_name,
                    "metric_value": r.metric_value,
                    "quality": r.quality,
                    "parameter_count": r.parameter_count,
                    "train_seconds": r.train_seconds,
                    "failure_reason": r.failure_reason,
                }
                for bid, r in benchmark_results.items()
            }
            for genome_id, benchmark_results in state.results.items()
        },
        "total_evaluations": state.total_evaluations,
    }
    path = os.path.join(run_dir, "checkpoints", f"gen_{generation:04d}.json")
    Path(path).write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    # Also write latest state
    latest = os.path.join(run_dir, "state.json")
    Path(latest).write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def _try_resume(run_dir: str) -> tuple[GenerationState | None, int]:
    """Attempt to load state from checkpoint. Returns (state, next_gen)."""
    from prism.runtime.training import EvaluationResult

    state_path = os.path.join(run_dir, "state.json")
    if not os.path.exists(state_path):
        return None, 0

    try:
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, 0

    population = [ModelGenome.model_validate(g) for g in data["population"]]

    results: dict[str, dict[str, EvaluationResult]] = {}
    for genome_id, benchmark_results in data.get("results", {}).items():
        results[genome_id] = {
            bid: EvaluationResult(**r_data)
            for bid, r_data in benchmark_results.items()
        }

    state = GenerationState(
        generation=data["generation"],
        population=population,
        results=results,
        total_evaluations=data.get("total_evaluations", 0),
        parent_ids={
            genome_id: list(parent_ids)
            for genome_id, parent_ids in data.get("parent_ids", {}).items()
        },
    )

    return state, data["generation"] + 1


def _write_summary(run_dir: str, state: GenerationState, elapsed: float) -> None:
    """Write a final summary.json with best results."""
    summaries = summaries_from_state_results(
        state.population, state.results, state.generation,
    )
    best = max(summaries, key=lambda s: s.aggregate_quality) if summaries else None

    summary = {
        "elapsed_seconds": elapsed,
        "total_evaluations": state.total_evaluations,
        "population_size": len(state.population),
        "best_genome_id": best.genome_id if best else None,
        "best_family": best.family if best else None,
        "best_quality": best.aggregate_quality if best else None,
        "best_parameter_count": best.parameter_count if best else None,
        "families_active": sorted({g.family for g in state.population}),
    }

    path = os.path.join(run_dir, "summary.json")
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _resolved_run_config(config: RunConfig, benchmark_specs: list) -> RunConfig:
    if config.evolution.allowed_families:
        return config

    allowed_sets: list[set[str]] = []
    for spec in benchmark_specs:
        modality = getattr(spec, "modality", "tabular")
        task = getattr(spec, "task", "classification")
        families = set(compatible_families(modality))
        if task == "language_modeling":
            families &= {"embedding", "attention", "sparse_attention"}
        if families:
            allowed_sets.append(families)

    if not allowed_sets:
        return config

    allowed = sorted(set.intersection(*allowed_sets))
    if not allowed:
        return config

    return config.model_copy(
        update={
            "evolution": config.evolution.model_copy(update={"allowed_families": allowed}),
        }
    )


def _persist_lineage(store: RunStore, run_id: str, generation: int, lineage: list[dict]) -> None:
    for record in lineage:
        parent_ids = record.get("parent_ids", [])
        operator = record.get("operator", "mutation")
        for parent_id in parent_ids or [None]:
            store.save_lineage(
                run_id,
                record["genome_id"],
                parent_id,
                generation,
                operator,
                operator_kind=operator.split(":", 1)[0],
            )


def _persist_archives(store: RunStore, run_id: str, archives: dict) -> None:
    for summary in archives.get("pareto", []):
        store.save_archive(run_id, "pareto", None, summary.genome_id, summary.aggregate_quality)

    for benchmark_id, elites in archives.get("elite", {}).items():
        for summary in elites:
            store.save_archive(run_id, "elite", benchmark_id, summary.genome_id, summary.aggregate_quality)

    for family, summary in archives.get("niche", {}).items():
        store.save_archive(run_id, f"niche:{family}", None, summary.genome_id, summary.aggregate_quality)
