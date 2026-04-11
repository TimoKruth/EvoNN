"""Evaluation stage: compile genomes, train models, return fitnesses."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from topograph.benchmarks.preprocess import Preprocessor
from topograph.benchmarks.spec import BenchmarkSpec
from topograph.cache import WeightCache, structural_hash
from topograph.config import RunConfig
from topograph.genome.genome import Genome
from topograph.nn.compiler import compile_genome, estimate_model_bytes
from topograph.nn.train import (
    compute_percentile_fitness,
    effective_model_bytes,
    extract_weights,
    load_weight_snapshot,
    train_model,
)
from topograph.parallel import ParallelEvaluator
from topograph.pipeline.archive import compute_behavior


@dataclass
class GenerationState:
    generation: int
    population: list[Genome]
    fitnesses: list[float] = field(default_factory=list)
    model_bytes: list[int] = field(default_factory=list)
    behaviors: list[np.ndarray] = field(default_factory=list)
    phase: str = "explore"
    total_evaluations: int = 0


def evaluate(
    state: GenerationState,
    config: RunConfig,
    benchmark_spec: BenchmarkSpec,
    cache: WeightCache | None = None,
    parallel_eval: ParallelEvaluator | None = None,
    multi_fidelity_schedule: list[float] | None = None,
) -> GenerationState:
    """Evaluate all genomes in population. Returns updated state with fitnesses."""
    tc = config.training

    # Load and preprocess data
    X_train, y_train, X_val, y_val = benchmark_spec.load_data(
        seed=config.seed, validation_split=tc.validation_split,
    )
    pp = Preprocessor()
    X_train = pp.fit_transform(X_train)
    X_val = pp.transform(X_val)

    input_dim = benchmark_spec.input_dim or X_train.shape[1]
    num_classes = benchmark_spec.num_classes or 1

    fitnesses: list[float] = []
    model_bytes_list: list[int] = []
    behaviors: list[np.ndarray] = []

    for genome in state.population:
        try:
            fitness, mb = _evaluate_single(
                genome=genome,
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_dim=input_dim,
                num_classes=num_classes,
                task=benchmark_spec.task,
                cache=cache,
                multi_fidelity_schedule=multi_fidelity_schedule,
                generation=state.generation,
            )
        except (ValueError, KeyError, RuntimeError):
            fitness = float("inf")
            mb = estimate_model_bytes(genome)

        fitnesses.append(fitness)
        model_bytes_list.append(mb)
        behaviors.append(compute_behavior(genome))

    state.fitnesses = fitnesses
    state.model_bytes = model_bytes_list
    state.behaviors = behaviors
    state.total_evaluations += len(state.population)
    return state


def _evaluate_single(
    genome: Genome,
    config: RunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    task: str,
    cache: WeightCache | None,
    multi_fidelity_schedule: list[float] | None,
    generation: int,
) -> tuple[float, int]:
    """Compile, optionally load cached weights, train, cache weights, return (fitness, model_bytes)."""
    tc = config.training

    model = compile_genome(
        genome,
        input_dim=input_dim,
        num_classes=num_classes,
        task=task,
        layer_norm=tc.layer_norm,
    )

    # Determine effective epochs
    base_epochs = tc.epochs
    lr = genome.learning_rate if genome.learning_rate is not None else tc.learning_rate
    batch_size = genome.batch_size if genome.batch_size is not None else tc.batch_size

    # Weight inheritance: check cache for exact or partial match
    cache_hit = False
    if cache is not None:
        exact = cache.lookup(genome)
        if exact is not None:
            loaded = load_weight_snapshot(model, exact)
            if loaded > 0:
                cache_hit = True
                base_epochs = max(1, int(base_epochs * tc.finetune_epoch_ratio))
        else:
            partial = cache.lookup_partial(genome)
            if partial is not None:
                loaded = load_weight_snapshot(model, partial)
                if loaded > 0:
                    base_epochs = max(1, int(base_epochs * tc.partial_epoch_ratio))

    # Multi-fidelity scaling
    epochs = base_epochs
    if multi_fidelity_schedule is not None and generation < len(multi_fidelity_schedule):
        scale = multi_fidelity_schedule[generation]
        epochs = max(1, int(epochs * scale))
    elif tc.multi_fidelity and multi_fidelity_schedule is None:
        # Default linear ramp: 30% at gen 0, 100% at final gen
        total_gens = config.evolution.num_generations
        if total_gens > 1:
            progress = generation / (total_gens - 1)
            scale = 0.3 + 0.7 * progress
            epochs = max(1, int(epochs * scale))

    # Train
    fitness = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        task=task,
        lr_schedule=tc.lr_schedule,
        weight_decay=tc.weight_decay,
        grad_clip_norm=tc.grad_clip_norm or 1.0,
    )

    # Store weights in cache
    if cache is not None and fitness != float("inf"):
        weights = extract_weights(model)
        cache.store(genome, weights)

    # Model size
    mb = effective_model_bytes(genome, input_dim=input_dim, num_classes=num_classes)

    # Update genome metadata
    genome.fitness = fitness
    genome.model_bytes = mb

    return fitness, mb


def evaluate_pool(
    state: GenerationState,
    config: RunConfig,
    benchmark_specs: list[BenchmarkSpec],
    cache: WeightCache | None = None,
    parallel_eval: ParallelEvaluator | None = None,
    rng: random.Random | None = None,
) -> GenerationState:
    """Multi-benchmark evaluation. Sample K benchmarks, aggregate via percentile."""
    pool_cfg = config.benchmark_pool
    if pool_cfg is None or not benchmark_specs:
        return state

    rng = rng or random.Random(config.seed)
    k = min(pool_cfg.sample_k, len(benchmark_specs))

    # Sample benchmarks (undercovered bias not applied here -- coordinator tracks that)
    sampled = rng.sample(benchmark_specs, k) if k < len(benchmark_specs) else list(benchmark_specs)

    tc = config.training
    raw_losses: dict[str, list[float]] = {}

    for spec in sampled:
        X_train, y_train, X_val, y_val = spec.load_data(
            seed=config.seed, validation_split=tc.validation_split,
        )
        pp = Preprocessor()
        X_train = pp.fit_transform(X_train)
        X_val = pp.transform(X_val)

        input_dim = spec.input_dim or X_train.shape[1]
        num_classes = spec.num_classes or 1

        losses: list[float] = []
        for genome in state.population:
            try:
                fitness, _ = _evaluate_single(
                    genome=genome,
                    config=config,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    task=spec.task,
                    cache=cache,
                    multi_fidelity_schedule=config.training.multi_fidelity_schedule,
                    generation=state.generation,
                )
            except (ValueError, KeyError, RuntimeError):
                fitness = float("inf")
            losses.append(fitness)

        raw_losses[spec.name] = losses

    # Aggregate via percentile fitness
    state.fitnesses = compute_percentile_fitness(raw_losses)
    state.model_bytes = [
        estimate_model_bytes(g) for g in state.population
    ]
    state.behaviors = [compute_behavior(g) for g in state.population]
    state.total_evaluations += len(state.population) * len(sampled)
    return state


def score(state: GenerationState, config: RunConfig) -> GenerationState:
    """Apply complexity penalty and device target constraints to raw fitnesses."""
    alpha = config.complexity_penalty
    target = config.target_device

    for i in range(len(state.fitnesses)):
        if state.fitnesses[i] == float("inf"):
            continue

        # Parsimony pressure
        if alpha > 0 and i < len(state.model_bytes):
            state.fitnesses[i] += alpha * state.model_bytes[i]

        # Device target hard penalty
        if target and target.max_model_bytes and i < len(state.model_bytes):
            if state.model_bytes[i] > target.max_model_bytes:
                state.fitnesses[i] += 10.0

    # Update genome fitness values
    for i, genome in enumerate(state.population):
        if i < len(state.fitnesses):
            genome.fitness = state.fitnesses[i]

    return state
