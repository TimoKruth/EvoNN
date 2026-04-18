"""Main evolution loop: setup, evaluate, reproduce, checkpoint."""

from __future__ import annotations

from collections import Counter
import math
import os
import random
import time
from dataclasses import asdict

from topograph.benchmarks.parity import get_benchmark, resolve_benchmark_pool_names
from topograph.benchmarks.spec import BenchmarkSpec
from topograph.cache import WeightCache
from topograph.config import RunConfig
from topograph.genome.codec import dict_to_genome, genome_to_dict
from topograph.genome.genes import OperatorType
from topograph.genome.genome import Genome, InnovationCounter
from topograph.monitor import TerminalMonitor
from topograph.parallel import ParallelEvaluator, ParallelRuntimeLimits
from topograph.pipeline.archive import (
    BenchmarkEliteArchive,
    MAPElitesArchive,
    NoveltyArchive,
)
from topograph.pipeline.evaluate import (
    BenchmarkDataCache,
    EvaluationMemo,
    GenerationState,
    evaluate,
    evaluate_pool,
    score,
)
from topograph.pipeline.reproduce import PendingMutationOutcome, reproduce
from topograph.pipeline.schedule import MutationScheduler
from topograph.storage import RunStore

_MAP_ELITES_TOTAL_NICHES = 6**4


def run_evolution(
    config: RunConfig,
    benchmark_spec: BenchmarkSpec | None = None,
    run_dir: str | None = None,
    resume: bool = False,
) -> GenerationState:
    """Run full evolution through the coordinator path."""
    rng = random.Random(config.seed)
    innovation_counter = InnovationCounter()
    run_id = "current"
    monitor = TerminalMonitor()
    scheduler = MutationScheduler(config.evolution.mutation_rates)

    store: RunStore | None = None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        store = RunStore(os.path.join(run_dir, "metrics.duckdb"))
        store.save_run(run_id, config.model_dump(mode="json"))

    cache = WeightCache() if config.training.weight_inheritance else None
    data_cache = BenchmarkDataCache()
    evaluation_memo = EvaluationMemo()
    parallel_eval = (
        ParallelEvaluator(
            config.training.parallel_workers,
            runtime_limits=_parallel_runtime_limits(config),
        )
        if config.training.parallel_workers != 1
        else None
    )

    novelty_archive = NoveltyArchive(
        max_size=config.novelty_archive_size,
        k=config.novelty_k,
    ) if config.novelty_weight > 0 else None
    map_elites_archive = MAPElitesArchive() if config.map_elites else None
    benchmark_elite_archive = BenchmarkEliteArchive() if config.benchmark_elite_archive else None

    benchmark_specs: list[BenchmarkSpec] | None = None
    if config.benchmark_pool:
        benchmark_specs = [
            get_benchmark(name) for name in resolve_benchmark_pool_names(config.benchmark_pool)
        ]
    elif benchmark_spec is None:
        benchmark_spec = get_benchmark(config.benchmark)

    state = GenerationState(generation=0, population=[])
    fitness_history: list[float] = []
    pending_outcomes: list[PendingMutationOutcome] = []
    pool_state = {
        "current_sample": [],
        "rotation_counter": 0,
        "benchmark_best_fitness": {},
    }
    elapsed_before_resume = 0.0
    novelty_score_sum = 0.0
    novelty_score_count = 0
    novelty_score_max = 0.0
    map_elites_insertions = 0
    start_gen = 0
    completed = False

    if resume and store:
        snapshot = store.load_run_state(run_id)
        if snapshot is not None:
            state = GenerationState(
                generation=int(snapshot.get("next_generation", 0)),
                population=[
                    dict_to_genome(genome_dict)
                    for genome_dict in snapshot.get("population", [])
                ],
                total_evaluations=int(snapshot.get("total_evaluations", 0)),
            )
            innovation_counter = InnovationCounter(
                int(snapshot.get("innovation_counter", innovation_counter.value))
            )
            start_gen = state.generation
            fitness_history = [float(x) for x in snapshot.get("fitness_history", [])]
            elapsed_before_resume = float(snapshot.get("elapsed_seconds", 0.0))
            novelty_score_sum = float(snapshot.get("novelty_score_sum", 0.0))
            novelty_score_count = int(snapshot.get("novelty_score_count", 0))
            novelty_score_max = float(snapshot.get("novelty_score_max", 0.0))
            map_elites_insertions = int(snapshot.get("map_elites_insertions", 0))
            pool_state = snapshot.get("pool_state", pool_state)
            scheduler.load_dict(snapshot.get("scheduler"))
            pending_outcomes = [
                PendingMutationOutcome(
                    genome_idx=int(item["genome_idx"]),
                    baseline_fitness=float(item["baseline_fitness"]),
                    operators=[str(op) for op in item["operators"]],
                )
                for item in snapshot.get("pending_outcomes", [])
            ]
            if novelty_archive is not None:
                novelty_archive = NoveltyArchive.from_dict(snapshot.get("novelty_archive"))
            if map_elites_archive is not None:
                map_elites_archive = MAPElitesArchive.from_dict(snapshot.get("map_elites_archive"))
            if benchmark_elite_archive is not None:
                benchmark_elite_archive = BenchmarkEliteArchive.from_dict(
                    snapshot.get("benchmark_elite_archive")
                )
            completed = bool(snapshot.get("completed", False))
            monitor.on_info(f"Resuming from generation {start_gen}")
        else:
            latest_gen = store.load_latest_generation(run_id)
            if latest_gen is not None:
                genome_dicts = store.load_genomes(run_id, latest_gen)
                state = GenerationState(
                    generation=latest_gen + 1,
                    population=[dict_to_genome(d) for d in genome_dicts],
                )
                counter_val = store.load_innovation_counter(run_id)
                if counter_val is not None:
                    innovation_counter = InnovationCounter(counter_val)
                start_gen = state.generation
                monitor.on_info(
                    "Legacy checkpoint detected; resuming without scheduler/QD state."
                )

    if start_gen >= config.evolution.num_generations:
        if state.population:
            if store is not None:
                store.close()
            return state
        latest_gen = store.load_latest_generation(run_id) if store else None
        if latest_gen is not None and store is not None:
            state.population = [
                dict_to_genome(d) for d in store.load_genomes(run_id, latest_gen)
            ]
            state.generation = latest_gen
            store.close()
        return state

    if not state.population:
        state = GenerationState(
            generation=0,
            population=_create_seed_population(config, innovation_counter, rng),
        )
        start_gen = 0

    run_start = time.time()
    completed_generations = start_gen

    try:
        for gen in range(start_gen, config.evolution.num_generations):
            state.generation = gen
            state.phase = scheduler.current_phase(gen, config.evolution.num_generations).value

            sampled_specs = benchmark_specs
            if benchmark_specs is not None and config.benchmark_pool is not None:
                sampled_specs = _sample_benchmark_specs(config, benchmark_specs, rng, pool_state)
                eval_pool_kwargs = {
                    "state": state,
                    "config": config,
                    "benchmark_specs": sampled_specs,
                    "cache": cache,
                    "data_cache": data_cache,
                    "evaluation_memo": evaluation_memo,
                    "progress_callback": lambda stage, payload, gen=gen: _benchmark_progress(
                        monitor, gen, stage, payload,
                    ),
                }
                if parallel_eval is not None:
                    eval_pool_kwargs["parallel_eval"] = parallel_eval
                state = evaluate_pool(**eval_pool_kwargs)
            else:
                eval_kwargs = {
                    "state": state,
                    "config": config,
                    "benchmark_spec": benchmark_spec,
                    "cache": cache,
                    "multi_fidelity_schedule": config.training.multi_fidelity_schedule,
                    "data_cache": data_cache,
                    "evaluation_memo": evaluation_memo,
                    "progress_callback": lambda stage, payload, gen=gen: _benchmark_progress(
                        monitor, gen, stage, payload,
                    ),
                }
                if parallel_eval is not None:
                    eval_kwargs["parallel_eval"] = parallel_eval
                state = evaluate(**eval_kwargs)

            state = score(state, config)

            novelty_scores: list[float] = []
            if novelty_archive is not None and state.behaviors:
                novelty_scores = _blend_novelty(state, config, novelty_archive)
                if novelty_scores:
                    novelty_score_sum += sum(novelty_scores)
                    novelty_score_count += len(novelty_scores)
                    novelty_score_max = max(novelty_score_max, max(novelty_scores))

            if pending_outcomes:
                _record_pending_outcomes(state, scheduler, pending_outcomes)
                pending_outcomes = []

            if sampled_specs is not None and state.raw_losses:
                _update_pool_fitness_history(pool_state, state.raw_losses)

            if map_elites_archive is not None:
                for i, genome in enumerate(state.population):
                    if i >= len(state.fitnesses) or i >= len(state.behaviors):
                        continue
                    if state.fitnesses[i] == float("inf"):
                        continue
                    if map_elites_archive.add(genome, state.behaviors[i], state.fitnesses[i]):
                        map_elites_insertions += 1

            if benchmark_elite_archive is not None:
                _update_benchmark_elites(
                    benchmark_elite_archive,
                    generation=gen,
                    raw_losses=state.raw_losses,
                )

            if state.fitnesses:
                finite = [f for f in state.fitnesses if f != float("inf")]
                if finite:
                    best_fit = min(finite)
                    avg_fit = sum(finite) / len(finite)
                    worst_fit = max(finite)
                    fitness_history.append(best_fit)
                    completed_generations = gen + 1
                    monitor.on_generation(
                        gen,
                        config.evolution.num_generations,
                        best_fitness=best_fit,
                        avg_fitness=avg_fit,
                        worst_fitness=worst_fit,
                        phase=state.phase,
                        population_size=len(state.population),
                        archive_fill=_archive_fill_ratio(map_elites_archive),
                        scheduler_stats=_generation_stats(
                            sampled_specs,
                            benchmark_elite_archive,
                            state,
                        ),
                    )

            if store:
                _checkpoint_generation(store, run_id, gen, state, innovation_counter)

            should_stop = (
                config.early_stopping is not None
                and config.early_stopping.enabled
                and _should_stop_early(fitness_history, config.early_stopping)
            )

            if should_stop:
                monitor.on_info(
                    f"Early stopping at generation {gen + 1}: "
                    f"no improvement over {config.early_stopping.window} generations"
                )
                break

            if gen < config.evolution.num_generations - 1:
                protected = (
                    benchmark_elite_archive.get_elite_indices()
                    if benchmark_elite_archive is not None
                    else set()
                )
                state, pending_outcomes = reproduce(
                    state,
                    config,
                    innovation_counter,
                    scheduler,
                    rng,
                    protected_indices=protected,
                )
                pending_outcomes = _inject_map_elites(
                    state,
                    pending_outcomes,
                    map_elites_archive,
                    config,
                    rng,
                )

                if store:
                    _save_resume_snapshot(
                        store=store,
                        run_id=run_id,
                        next_generation=gen + 1,
                        state=state,
                        innovation_counter=innovation_counter,
                        fitness_history=fitness_history,
                        scheduler=scheduler,
                        novelty_archive=novelty_archive,
                        map_elites_archive=map_elites_archive,
                        benchmark_elite_archive=benchmark_elite_archive,
                        pending_outcomes=pending_outcomes,
                        pool_state=pool_state,
                        elapsed_seconds=elapsed_before_resume + (time.time() - run_start),
                        total_evaluations=state.total_evaluations,
                        novelty_score_sum=novelty_score_sum,
                        novelty_score_count=novelty_score_count,
                        novelty_score_max=novelty_score_max,
                        map_elites_insertions=map_elites_insertions,
                        completed=False,
                    )

        elapsed = elapsed_before_resume + (time.time() - run_start)
        best_fit = min(state.fitnesses) if state.fitnesses else float("inf")
        monitor.on_complete(best_fit, completed_generations, elapsed=elapsed)

        if store:
            _save_budget_metadata(
                store=store,
                run_id=run_id,
                state=state,
                config=config,
                completed_generations=completed_generations,
                elapsed=elapsed,
                novelty_archive=novelty_archive,
                novelty_score_sum=novelty_score_sum,
                novelty_score_count=novelty_score_count,
                novelty_score_max=novelty_score_max,
                map_elites_archive=map_elites_archive,
                map_elites_insertions=map_elites_insertions,
                benchmark_elite_archive=benchmark_elite_archive,
                scheduler=scheduler,
                parallel_eval=parallel_eval,
            )
            _save_resume_snapshot(
                store=store,
                run_id=run_id,
                next_generation=completed_generations,
                state=state,
                innovation_counter=innovation_counter,
                fitness_history=fitness_history,
                scheduler=scheduler,
                novelty_archive=novelty_archive,
                map_elites_archive=map_elites_archive,
                benchmark_elite_archive=benchmark_elite_archive,
                pending_outcomes=[],
                pool_state=pool_state,
                elapsed_seconds=elapsed,
                total_evaluations=state.total_evaluations,
                novelty_score_sum=novelty_score_sum,
                novelty_score_count=novelty_score_count,
                novelty_score_max=novelty_score_max,
                map_elites_insertions=map_elites_insertions,
                completed=True,
            )
        return state
    except KeyboardInterrupt:
        elapsed = elapsed_before_resume + (time.time() - run_start)
        if store:
            _save_resume_snapshot(
                store=store,
                run_id=run_id,
                next_generation=state.generation,
                state=state,
                innovation_counter=innovation_counter,
                fitness_history=fitness_history,
                scheduler=scheduler,
                novelty_archive=novelty_archive,
                map_elites_archive=map_elites_archive,
                benchmark_elite_archive=benchmark_elite_archive,
                pending_outcomes=pending_outcomes,
                pool_state=pool_state,
                elapsed_seconds=elapsed,
                total_evaluations=state.total_evaluations,
                novelty_score_sum=novelty_score_sum,
                novelty_score_count=novelty_score_count,
                novelty_score_max=novelty_score_max,
                map_elites_insertions=map_elites_insertions,
                completed=False,
            )
            monitor.on_info(
                f"Interrupted at generation {state.generation}; resume snapshot saved."
            )
        raise
    finally:
        if parallel_eval is not None:
            parallel_eval.close()
        if store is not None:
            store.close()


def _create_seed_population(
    config: RunConfig,
    innovation_counter: InnovationCounter,
    rng: random.Random,
) -> list[Genome]:
    population = [
        Genome.create_seed(
            innovation_counter,
            random.Random(rng.randint(0, 2**32 - 1)),
            mixed_precision=bool(config.quantization_schedule),
        )
        for _ in range(config.evolution.population_size)
    ]

    diverse_ops = [OperatorType.RESIDUAL, OperatorType.ATTENTION_LITE]
    for i, op in enumerate(diverse_ops):
        if i >= len(population):
            break
        genome = population[i]
        for j, layer in enumerate(genome.layers):
            if not layer.enabled:
                continue
            update: dict[str, object] = {"operator": op}
            if op in (OperatorType.ATTENTION_LITE, OperatorType.TRANSFORMER_LITE):
                num_heads = max(1, layer.num_heads)
                while num_heads > 1 and layer.width % num_heads != 0:
                    num_heads -= 1
                update["num_heads"] = num_heads
            genome.layers[j] = layer.model_copy(update=update)

    for genome in population:
        genome.learning_rate = _sample_learning_rate(config, rng)
        genome.batch_size = _sample_batch_size(config, rng)
    return population


def _sample_learning_rate(config: RunConfig, rng: random.Random) -> float:
    base = config.training.learning_rate or 0.001
    sampled = base * (10 ** rng.uniform(-0.5, 0.5))
    return max(1e-5, min(1e-1, sampled))


def _sample_batch_size(config: RunConfig, rng: random.Random) -> int:
    base = config.training.batch_size or 32
    candidates = sorted({16, 32, 64, 128, base, max(8, base // 2), base * 2})
    return int(rng.choice(candidates))


def _sample_benchmark_specs(
    config: RunConfig,
    benchmark_specs: list[BenchmarkSpec],
    rng: random.Random,
    pool_state: dict[str, object],
) -> list[BenchmarkSpec]:
    pool_cfg = config.benchmark_pool
    if pool_cfg is None or not benchmark_specs:
        return benchmark_specs

    current_names = list(pool_state.get("current_sample", []))
    rotation_counter = int(pool_state.get("rotation_counter", 0))
    if pool_cfg.rotation_interval and current_names and rotation_counter < pool_cfg.rotation_interval:
        pool_state["rotation_counter"] = rotation_counter + 1
        by_name = {spec.name: spec for spec in benchmark_specs}
        return [by_name[name] for name in current_names if name in by_name]

    k = min(pool_cfg.sample_k, len(benchmark_specs))
    best_fitness = pool_state.get("benchmark_best_fitness", {})
    ranked = sorted(
        (
            (name, float(loss))
            for name, loss in best_fitness.items()
            if math.isfinite(float(loss))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    undercovered = {name for name, _ in ranked[: max(1, len(ranked) // 2)]}

    available = list(benchmark_specs)
    sample: list[BenchmarkSpec] = []
    while available and len(sample) < k:
        undercovered_available = [spec for spec in available if spec.name in undercovered]
        if (
            pool_cfg.undercovered_benchmark_bias > 0
            and undercovered_available
            and rng.random() < pool_cfg.undercovered_benchmark_bias
        ):
            chosen = rng.choice(undercovered_available)
        else:
            chosen = rng.choice(available)
        sample.append(chosen)
        available.remove(chosen)

    pool_state["current_sample"] = [spec.name for spec in sample]
    pool_state["rotation_counter"] = 1
    return sample


def _update_pool_fitness_history(
    pool_state: dict[str, object],
    raw_losses: dict[str, list[float]],
) -> None:
    best_fitness = dict(pool_state.get("benchmark_best_fitness", {}))
    for benchmark_name, losses in raw_losses.items():
        finite = [loss for loss in losses if math.isfinite(loss)]
        if not finite:
            continue
        current = min(finite)
        previous = float(best_fitness.get(benchmark_name, float("inf")))
        best_fitness[benchmark_name] = min(previous, current)
    pool_state["benchmark_best_fitness"] = best_fitness


def _blend_novelty(
    state: GenerationState,
    config: RunConfig,
    archive: NoveltyArchive,
) -> list[float]:
    """Blend novelty scores into fitnesses and add behaviors to archive."""
    lam = config.novelty_weight
    novelty_scores: list[float] = []
    for i, _genome in enumerate(state.population):
        if i >= len(state.fitnesses) or state.fitnesses[i] == float("inf"):
            continue
        if i >= len(state.behaviors):
            continue

        novelty = archive.compute_novelty(
            state.behaviors[i],
            population_behaviors=state.behaviors,
        )
        state.fitnesses[i] -= lam * novelty
        archive.add(state.behaviors[i])
        novelty_scores.append(novelty)
    return novelty_scores


def _update_benchmark_elites(
    archive: BenchmarkEliteArchive,
    *,
    generation: int,
    raw_losses: dict[str, list[float]],
) -> None:
    for benchmark_name, losses in raw_losses.items():
        for genome_idx, loss in enumerate(losses):
            if not math.isfinite(loss):
                continue
            archive.update(benchmark_name, genome_idx, float(loss), generation)


def _record_pending_outcomes(
    state: GenerationState,
    scheduler: MutationScheduler,
    pending_outcomes: list[PendingMutationOutcome],
) -> None:
    for outcome in pending_outcomes:
        if outcome.genome_idx >= len(state.fitnesses):
            continue
        current = state.fitnesses[outcome.genome_idx]
        if not math.isfinite(current) or not math.isfinite(outcome.baseline_fitness):
            improved = False
        else:
            improved = current < outcome.baseline_fitness
        for operator_name in outcome.operators:
            scheduler.record_outcome(operator_name, improved)


def _inject_map_elites(
    state: GenerationState,
    pending_outcomes: list[PendingMutationOutcome],
    archive: MAPElitesArchive | None,
    config: RunConfig,
    rng: random.Random,
) -> list[PendingMutationOutcome]:
    if archive is None or len(archive) == 0:
        return pending_outcomes

    inject_count = min(2, len(archive))
    injected = archive.sample(inject_count, rng)
    replaced_indices: set[int] = set()
    for offset, genome in enumerate(injected):
        idx = config.evolution.elite_count + offset
        if idx >= len(state.population):
            break
        genome.learning_rate = _sample_learning_rate(config, rng)
        genome.batch_size = _sample_batch_size(config, rng)
        genome.fitness = None
        genome.param_count = 0
        genome.model_bytes = 0
        state.population[idx] = genome
        replaced_indices.add(idx)

    if not replaced_indices:
        return pending_outcomes
    return [
        outcome for outcome in pending_outcomes if outcome.genome_idx not in replaced_indices
    ]


def _checkpoint_generation(
    store: RunStore,
    run_id: str,
    generation: int,
    state: GenerationState,
    innovation_counter: InnovationCounter,
) -> None:
    store.save_genomes(
        run_id,
        generation,
        [genome_to_dict(genome) for genome in state.population],
    )
    store.save_innovation_counter(run_id, innovation_counter.value)
    if state.benchmark_results:
        store.save_benchmark_results(run_id, generation, state.benchmark_results)
    if state.benchmark_timings:
        store.save_benchmark_timings(run_id, generation, state.benchmark_timings)


def _save_resume_snapshot(
    *,
    store: RunStore,
    run_id: str,
    next_generation: int,
    state: GenerationState,
    innovation_counter: InnovationCounter,
    fitness_history: list[float],
    scheduler: MutationScheduler,
    novelty_archive: NoveltyArchive | None,
    map_elites_archive: MAPElitesArchive | None,
    benchmark_elite_archive: BenchmarkEliteArchive | None,
    pending_outcomes: list[PendingMutationOutcome],
    pool_state: dict[str, object],
    elapsed_seconds: float,
    total_evaluations: int,
    novelty_score_sum: float,
    novelty_score_count: int,
    novelty_score_max: float,
    map_elites_insertions: int,
    completed: bool,
) -> None:
    store.save_run_state(
        run_id,
        {
            "next_generation": next_generation,
            "population": [genome_to_dict(genome) for genome in state.population],
            "innovation_counter": innovation_counter.value,
            "fitness_history": fitness_history,
            "scheduler": scheduler.to_dict(),
            "novelty_archive": novelty_archive.to_dict() if novelty_archive is not None else None,
            "map_elites_archive": (
                map_elites_archive.to_dict() if map_elites_archive is not None else None
            ),
            "benchmark_elite_archive": (
                benchmark_elite_archive.to_dict()
                if benchmark_elite_archive is not None
                else None
            ),
            "pending_outcomes": [asdict(item) for item in pending_outcomes],
            "pool_state": pool_state,
            "elapsed_seconds": elapsed_seconds,
            "total_evaluations": total_evaluations,
            "novelty_score_sum": novelty_score_sum,
            "novelty_score_count": novelty_score_count,
            "novelty_score_max": novelty_score_max,
            "map_elites_insertions": map_elites_insertions,
            "completed": completed,
        },
    )


def _save_budget_metadata(
    *,
    store: RunStore,
    run_id: str,
    state: GenerationState,
    config: RunConfig,
    completed_generations: int,
    elapsed: float,
    novelty_archive: NoveltyArchive | None,
    novelty_score_sum: float,
    novelty_score_count: int,
    novelty_score_max: float,
    map_elites_archive: MAPElitesArchive | None,
    map_elites_insertions: int,
    benchmark_elite_archive: BenchmarkEliteArchive | None,
    scheduler: MutationScheduler,
    parallel_eval: ParallelEvaluator | None = None,
) -> None:
    timing_rows = store.load_benchmark_timings(run_id)
    benchmark_results = store.load_benchmark_results(run_id)
    total_timing_seconds = sum(float(row["total_seconds"]) for row in timing_rows)
    reused_count = sum(int(row["reused_count"]) for row in timing_rows)
    trained_count = sum(int(row["trained_count"]) for row in timing_rows)
    failed_count = sum(int(row["failed_count"]) for row in timing_rows)
    data_cache_hits = sum(int(row.get("data_cache_hits", 0)) for row in timing_rows)
    data_cache_misses = sum(int(row.get("data_cache_misses", 0)) for row in timing_rows)
    max_resolved_workers = max(
        (int(row["resolved_worker_count"]) for row in timing_rows),
        default=1,
    )
    occupied = len(map_elites_archive) if map_elites_archive is not None else 0
    metadata = {
        "evaluation_count": state.total_evaluations,
        "wall_clock_seconds": round(elapsed, 2),
        "total_generations": completed_generations,
        "population_size": config.evolution.population_size,
        "benchmark_count": len(state.raw_losses) if state.raw_losses else 1,
        "novelty_archive_final_size": len(novelty_archive) if novelty_archive is not None else 0,
        "novelty_score_mean": (
            round(novelty_score_sum / novelty_score_count, 6)
            if novelty_score_count > 0
            else None
        ),
        "novelty_score_max": round(novelty_score_max, 6) if novelty_score_count > 0 else None,
        "map_elites_occupied_niches": occupied,
        "map_elites_total_niches": _MAP_ELITES_TOTAL_NICHES,
        "map_elites_fill_ratio": (
            round(occupied / _MAP_ELITES_TOTAL_NICHES, 6)
            if map_elites_archive is not None
            else None
        ),
        "map_elites_insertions": map_elites_insertions,
        "benchmark_elites": (
            len(benchmark_elite_archive.elites)
            if benchmark_elite_archive is not None
            else 0
        ),
        "cache_reused_count": reused_count,
        "cache_trained_count": trained_count,
        "cache_failed_count": failed_count,
        "data_cache_hits": data_cache_hits,
        "data_cache_misses": data_cache_misses,
        "cache_reuse_rate": (
            round(reused_count / max(reused_count + trained_count, 1), 6)
            if (reused_count + trained_count) > 0
            else None
        ),
        "evals_per_second": (
            round(state.total_evaluations / elapsed, 6) if elapsed > 0 else None
        ),
        "seconds_per_eval": (
            round(elapsed / state.total_evaluations, 6) if state.total_evaluations > 0 else None
        ),
        "benchmark_total_seconds": round(total_timing_seconds, 6),
        "requested_parallel_workers": (
            parallel_eval.requested_workers if parallel_eval is not None else 1
        ),
        "resolved_parallel_workers_max": max_resolved_workers,
        "worker_clamp_reason_counts": _worker_clamp_reason_counts(timing_rows),
        "sampled_benchmark_order_by_generation": _sampled_benchmark_order_by_generation(
            timing_rows
        ),
        "worst_benchmark_trend": _worst_benchmark_trend(benchmark_results),
        "operator_stats": scheduler.stats_summary(),
    }
    store.save_budget_metadata(run_id, metadata)


def _worker_clamp_reason_counts(timing_rows: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(
        str(row.get("worker_clamp_reason", "sequential"))
        for row in timing_rows
    )
    return dict(sorted(counts.items()))


def _sampled_benchmark_order_by_generation(
    timing_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_generation: dict[int, list[tuple[int, str]]] = {}
    for row in timing_rows:
        generation = int(row["generation"])
        by_generation.setdefault(generation, []).append(
            (int(row["benchmark_order"]), str(row["benchmark_name"]))
        )
    return [
        {
            "generation": generation,
            "benchmarks": [name for _, name in sorted(entries)],
        }
        for generation, entries in sorted(by_generation.items())
    ]


def _worst_benchmark_trend(
    benchmark_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_generation: dict[int, list[dict[str, object]]] = {}
    for row in benchmark_results:
        generation = int(row["generation"])
        by_generation.setdefault(generation, []).append(row)

    trend: list[dict[str, object]] = []
    for generation, rows in sorted(by_generation.items()):
        ok_rows = [
            row for row in rows
            if row.get("status") == "ok" and row.get("quality") is not None
        ]
        if not ok_rows:
            continue
        worst = min(ok_rows, key=lambda row: float(row["quality"]))
        trend.append(
            {
                "generation": generation,
                "benchmark_name": str(worst["benchmark_name"]),
                "quality": float(worst["quality"]),
                "metric_name": str(worst["metric_name"]),
                "metric_value": worst["metric_value"],
            }
        )
    return trend


def _archive_fill_ratio(archive: MAPElitesArchive | None) -> float | None:
    if archive is None:
        return None
    return len(archive) / _MAP_ELITES_TOTAL_NICHES


def _generation_stats(
    sampled_specs: list[BenchmarkSpec] | None,
    benchmark_elite_archive: BenchmarkEliteArchive | None,
    state: GenerationState | None = None,
) -> dict[str, object]:
    stats: dict[str, object] = {}
    if sampled_specs is not None:
        stats["Benchmarks"] = ", ".join(spec.name for spec in sampled_specs)
    if benchmark_elite_archive is not None:
        stats["Benchmark Elites"] = len(benchmark_elite_archive.elites)
    if state is not None and state.benchmark_timings:
        total_seconds = sum(float(item["total_seconds"]) for item in state.benchmark_timings)
        stats["Bench Time"] = f"{total_seconds:.1f}s"
        stats["Cache Reuse"] = f"{state.cache_reused}/{state.cache_reused + state.cache_trained}"
    return stats


def _parallel_runtime_limits(config: RunConfig) -> ParallelRuntimeLimits:
    tc = config.training
    return ParallelRuntimeLimits(
        cpu_fraction_limit=tc.parallel_cpu_fraction_limit,
        memory_fraction_limit=tc.parallel_memory_fraction_limit,
        reserved_system_memory_bytes=tc.parallel_reserved_system_memory_bytes,
        worker_thread_limit=tc.parallel_worker_thread_limit,
    )


def _benchmark_progress(
    monitor: TerminalMonitor,
    generation: int,
    stage: str,
    payload: dict[str, object],
) -> None:
    if stage == "start":
        monitor.on_benchmark_start(
            generation=generation,
            benchmark_name=str(payload["benchmark_name"]),
            benchmark_order=int(payload["benchmark_order"]),
            benchmark_total=int(payload["benchmark_total"]),
            task=str(payload["task"]),
        )
        return
    monitor.on_benchmark_complete(
        generation=generation,
        benchmark_name=str(payload["benchmark_name"]),
        benchmark_order=int(payload["benchmark_order"]),
        benchmark_total=int(payload["benchmark_total"]),
        total_seconds=float(payload["total_seconds"]),
        data_load_seconds=float(payload["data_load_seconds"]),
        evaluation_seconds=float(payload["evaluation_seconds"]),
        reused_count=int(payload["reused_count"]),
        trained_count=int(payload["trained_count"]),
        failed_count=int(payload["failed_count"]),
        resolved_worker_count=int(payload["resolved_worker_count"]),
    )


def _should_stop_early(
    fitness_history: list[float],
    es_config,
) -> bool:
    """Check if evolution has stagnated."""
    window = es_config.window
    threshold = es_config.threshold

    if len(fitness_history) < window:
        return False

    recent = fitness_history[-window:]
    improvement = recent[0] - recent[-1]
    return improvement < threshold
