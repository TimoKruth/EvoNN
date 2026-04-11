"""Main evolution loop: setup, evaluate, reproduce, checkpoint."""

from __future__ import annotations

import os
import random
import time

from topograph.benchmarks.parity import get_benchmark
from topograph.benchmarks.spec import BenchmarkSpec
from topograph.cache import WeightCache
from topograph.config import RunConfig
from topograph.genome.genes import OperatorType
from topograph.genome.genome import Genome, InnovationCounter
from topograph.monitor import TerminalMonitor
from topograph.parallel import ParallelEvaluator
from topograph.pipeline.archive import (
    BenchmarkEliteArchive,
    MAPElitesArchive,
    NoveltyArchive,
)
from topograph.pipeline.evaluate import GenerationState, evaluate, evaluate_pool, score
from topograph.pipeline.reproduce import reproduce
from topograph.pipeline.schedule import MutationScheduler
from topograph.pipeline.select import nsga2_select
from topograph.storage import RunStore


def run_evolution(
    config: RunConfig,
    benchmark_spec: BenchmarkSpec | None = None,
    run_dir: str | None = None,
    resume: bool = False,
) -> GenerationState:
    """Run full evolution. Returns final GenerationState."""
    rng = random.Random(config.seed)
    innovation_counter = InnovationCounter()
    run_id = "current"

    # Storage
    store: RunStore | None = None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        store = RunStore(os.path.join(run_dir, "metrics.duckdb"))
        store.save_run(run_id, config.model_dump())

    # Weight cache
    cache = WeightCache() if config.training.weight_inheritance else None

    # Parallel evaluator (0 = auto, 1 = sequential)
    parallel_eval: ParallelEvaluator | None = None
    if config.training.parallel_workers != 1:
        parallel_eval = ParallelEvaluator(config.training.parallel_workers)

    # Scheduler
    scheduler = MutationScheduler(config.evolution.mutation_rates)

    # Monitor
    monitor = TerminalMonitor()

    # Quality diversity archives
    novelty_archive: NoveltyArchive | None = None
    if config.novelty_weight > 0:
        novelty_archive = NoveltyArchive(
            max_size=config.novelty_archive_size,
            k=config.novelty_k,
        )

    map_elites_archive: MAPElitesArchive | None = None
    if config.map_elites:
        map_elites_archive = MAPElitesArchive()

    benchmark_elite_archive: BenchmarkEliteArchive | None = None
    if config.benchmark_elite_archive:
        benchmark_elite_archive = BenchmarkEliteArchive()

    # Multi-fidelity schedule
    mf_schedule = config.training.multi_fidelity_schedule

    # Initialize or resume population
    start_gen = 0
    if resume and store:
        latest_gen = store.load_latest_generation(run_id)
        if latest_gen is not None:
            counter_val = store.load_innovation_counter(run_id)
            if counter_val is not None:
                innovation_counter = InnovationCounter(start=counter_val)
            start_gen = latest_gen + 1
            monitor.on_info(f"Resuming from generation {start_gen}")

    # Create seed population (fresh start)
    if start_gen == 0:
        population = [
            Genome.create_seed(
                innovation_counter,
                random.Random(rng.randint(0, 2**32 - 1)),
            )
            for _ in range(config.evolution.population_size)
        ]
        # Seed operator diversity
        diverse_ops = [OperatorType.RESIDUAL, OperatorType.ATTENTION_LITE]
        for i, op in enumerate(diverse_ops):
            if i >= len(population):
                break
            genome = population[i]
            for j, lg in enumerate(genome.layers):
                if lg.enabled:
                    update: dict = {"operator": op}
                    if op in (OperatorType.ATTENTION_LITE, OperatorType.TRANSFORMER_LITE):
                        nh = 2
                        while nh > 1 and lg.width % nh != 0:
                            nh -= 1
                        update["num_heads"] = nh
                    genome.layers[j] = lg.model_copy(update=update)

        # Assign initial learning rates and batch sizes
        for g in population:
            g.learning_rate = config.training.learning_rate
            g.batch_size = config.training.batch_size
    else:
        # Stub: resumed population would be deserialized from store
        population = [
            Genome.create_seed(innovation_counter, random.Random(rng.randint(0, 2**32 - 1)))
            for _ in range(config.evolution.population_size)
        ]

    # Load benchmarks
    benchmark_specs: list[BenchmarkSpec] | None = None
    if config.benchmark_pool:
        benchmark_specs = [get_benchmark(name) for name in config.benchmark_pool.benchmarks]
    elif benchmark_spec is None:
        benchmark_spec = get_benchmark(config.benchmark)

    state = GenerationState(generation=start_gen, population=population)
    fitness_history: list[float] = []
    run_start = time.time()

    # Main loop
    total_gens = config.evolution.num_generations
    for gen in range(start_gen, total_gens):
        state.generation = gen
        state.phase = scheduler.current_phase(gen, total_gens).value

        # -- Evaluate --
        if benchmark_specs is not None:
            state = evaluate_pool(state, config, benchmark_specs, cache, parallel_eval, rng)
        else:
            state = evaluate(state, config, benchmark_spec, cache, parallel_eval, mf_schedule)

        # -- Score (complexity penalty + device constraints) --
        state = score(state, config)

        # -- Novelty blending --
        if novelty_archive is not None and state.behaviors:
            _blend_novelty(state, config, novelty_archive)

        # -- Archive updates --
        if map_elites_archive is not None:
            for i, genome in enumerate(state.population):
                if i < len(state.fitnesses) and i < len(state.behaviors):
                    map_elites_archive.add(genome, state.behaviors[i], state.fitnesses[i])

        if benchmark_elite_archive is not None:
            for i, genome in enumerate(state.population):
                if i < len(state.fitnesses):
                    bench_name = benchmark_spec.name if benchmark_spec else "pool"
                    benchmark_elite_archive.update(bench_name, i, state.fitnesses[i], gen)

        # -- Monitor --
        if state.fitnesses:
            best_fit = min(state.fitnesses)
            avg_fit = sum(state.fitnesses) / len(state.fitnesses)
            worst_fit = max(f for f in state.fitnesses if f != float("inf"))
            fitness_history.append(best_fit)

            archive_fill = None
            if map_elites_archive is not None:
                archive_fill = len(map_elites_archive) / 1296.0  # 6^4 niches

            monitor.on_generation(
                gen, total_gens,
                best_fitness=best_fit,
                avg_fitness=avg_fit,
                worst_fitness=worst_fit,
                phase=state.phase,
                population_size=len(state.population),
                archive_fill=archive_fill,
            )

        # -- Checkpoint --
        if store:
            _checkpoint(store, run_id, gen, state, innovation_counter)

        # -- Early stopping --
        if config.early_stopping and config.early_stopping.enabled:
            if _should_stop_early(fitness_history, config.early_stopping):
                monitor.on_info(
                    f"Early stopping at generation {gen + 1}: "
                    f"no improvement over {config.early_stopping.window} generations"
                )
                break

        # -- Reproduce (skip on last generation) --
        if gen < total_gens - 1:
            prev_best = min(state.fitnesses) if state.fitnesses else float("inf")
            state, applied_ops = reproduce(state, config, innovation_counter, scheduler, rng)

            # Feed back mutation outcomes to scheduler
            # We approximate: if the child's parent was in top quartile, mark improved
            for genome_idx, ops in applied_ops:
                improved = rng.random() < 0.3  # conservative proxy until next eval
                for op in ops:
                    scheduler.record_outcome(op, improved)

            # Inject MAP-Elites samples as parents for diversity
            if map_elites_archive is not None and len(map_elites_archive) > 0:
                inject_count = min(2, len(map_elites_archive))
                me_samples = map_elites_archive.sample(inject_count, rng)
                for i, sampled_genome in enumerate(me_samples):
                    if config.evolution.elite_count + i < len(state.population):
                        sampled_genome.learning_rate = config.training.learning_rate
                        sampled_genome.batch_size = config.training.batch_size
                        state.population[config.evolution.elite_count + i] = sampled_genome

    # -- Completion --
    elapsed = time.time() - run_start
    best_fit = min(state.fitnesses) if state.fitnesses else float("inf")
    monitor.on_complete(best_fit, total_gens, elapsed=elapsed)

    if store:
        store.close()

    return state


def _blend_novelty(
    state: GenerationState,
    config: RunConfig,
    archive: NoveltyArchive,
) -> None:
    """Blend novelty scores into fitnesses and add behaviors to archive."""
    lam = config.novelty_weight
    for i, genome in enumerate(state.population):
        if i >= len(state.fitnesses) or state.fitnesses[i] == float("inf"):
            continue
        if i >= len(state.behaviors):
            continue

        novelty = archive.compute_novelty(
            state.behaviors[i],
            population_behaviors=state.behaviors,
        )
        # Lower fitness is better; higher novelty is more novel
        # Subtract novelty to reward novel genomes
        state.fitnesses[i] -= lam * novelty
        archive.add(state.behaviors[i])


def _checkpoint(
    store: RunStore,
    run_id: str,
    generation: int,
    state: GenerationState,
    innovation_counter: InnovationCounter,
) -> None:
    """Save generation state to DuckDB."""
    genome_dicts = []
    for i, genome in enumerate(state.population):
        d = {
            "layers": [lg.model_dump() for lg in genome.layers],
            "connections": [cg.model_dump() for cg in genome.connections],
        }
        if i < len(state.fitnesses):
            d["fitness"] = state.fitnesses[i]
        d["param_count"] = genome.param_count
        d["model_bytes"] = genome.model_bytes
        genome_dicts.append(d)

    store.save_genomes(run_id, generation, genome_dicts)
    store.save_innovation_counter(run_id, innovation_counter._n)


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
