"""Evaluation stage: compile genomes, train models, return fitnesses."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from topograph.benchmarks.preprocess import Preprocessor
from topograph.benchmarks.spec import BenchmarkSpec
from topograph.cache import WeightCache, structural_hash
from topograph.config import RunConfig
from topograph.genome.genome import Genome
from topograph.nn.compiler import compile_genome, estimate_model_bytes
from topograph.nn.train import (
    EvaluationResult,
    compute_percentile_fitness,
    effective_model_bytes,
    extract_weights,
    load_weight_snapshot,
    train_and_evaluate,
)
from topograph.parallel import ParallelBatchContext, ParallelEvaluator, ParallelGenomeTask
from topograph.pipeline.archive import compute_behavior


@dataclass
class GenerationState:
    generation: int
    population: list[Genome]
    fitnesses: list[float] = field(default_factory=list)
    model_bytes: list[int] = field(default_factory=list)
    behaviors: list[np.ndarray] = field(default_factory=list)
    benchmark_results: list[dict] = field(default_factory=list)
    benchmark_timings: list[dict] = field(default_factory=list)
    raw_losses: dict[str, list[float]] = field(default_factory=dict)
    phase: str = "explore"
    total_evaluations: int = 0
    cache_reused: int = 0
    cache_trained: int = 0
    cache_failed: int = 0


class BenchmarkDataCache:
    """In-memory cache of loaded and preprocessed benchmark arrays."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, float], tuple[np.ndarray, ...]] = {}

    def get(
        self,
        benchmark_spec: BenchmarkSpec,
        *,
        seed: int,
        validation_split: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (benchmark_spec.name, seed, validation_split)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        X_train, y_train, X_val, y_val = benchmark_spec.load_data(
            seed=seed, validation_split=validation_split,
        )

        if benchmark_spec.task == "language_modeling":
            value = (
                X_train.astype(np.int32, copy=False),
                y_train.astype(np.int64, copy=False),
                X_val.astype(np.int32, copy=False),
                y_val.astype(np.int64, copy=False),
            )
        else:
            pp = Preprocessor()
            value = (
                pp.fit_transform(X_train),
                y_train,
                pp.transform(X_val),
                y_val,
            )

        self._cache[key] = value
        return value


class EvaluationMemo:
    """Run-local exact evaluation memoization keyed by benchmark and genome structure."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, int, str, int], tuple[EvaluationResult, int]] = {}

    def lookup(
        self,
        *,
        benchmark_name: str,
        genome: Genome,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> tuple[EvaluationResult, int] | None:
        entry = self._cache.get(
            self._key(
                benchmark_name=benchmark_name,
                genome=genome,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
        )
        if entry is None:
            return None
        return copy.deepcopy(entry[0]), entry[1]

    def store(
        self,
        *,
        benchmark_name: str,
        genome: Genome,
        epochs: int,
        lr: float,
        batch_size: int,
        result: EvaluationResult,
        model_bytes: int,
    ) -> None:
        self._cache[
            self._key(
                benchmark_name=benchmark_name,
                genome=genome,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
        ] = (copy.deepcopy(result), model_bytes)

    @staticmethod
    def _key(
        *,
        benchmark_name: str,
        genome: Genome,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> tuple[str, str, int, str, int]:
        return (
            benchmark_name,
            structural_hash(genome),
            epochs,
            f"{float(lr):.12g}",
            int(batch_size),
        )


def evaluate(
    state: GenerationState,
    config: RunConfig,
    benchmark_spec: BenchmarkSpec,
    cache: WeightCache | None = None,
    multi_fidelity_schedule: list[float] | None = None,
    data_cache: BenchmarkDataCache | None = None,
    evaluation_memo: EvaluationMemo | None = None,
    parallel_eval: ParallelEvaluator | None = None,
    progress_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> GenerationState:
    """Evaluate all genomes in population. Returns updated state with fitnesses."""
    tc = config.training

    if progress_callback is not None:
        progress_callback(
            "start",
            {
                "benchmark_name": benchmark_spec.name,
                "benchmark_order": 0,
                "benchmark_total": 1,
                "task": benchmark_spec.task,
            },
        )

    # Load and preprocess data
    load_start = time.perf_counter()
    X_train, y_train, X_val, y_val = _prepare_benchmark_data(
        benchmark_spec,
        seed=config.seed,
        validation_split=tc.validation_split,
        data_cache=data_cache,
    )
    data_load_seconds = time.perf_counter() - load_start

    input_dim = benchmark_spec.input_dim or X_train.shape[1]
    num_classes = _resolve_model_output_dim(
        benchmark_spec=benchmark_spec,
        X_train=X_train,
        y_train=y_train,
    )

    fitnesses: list[float] = []
    model_bytes_list: list[int] = []
    behaviors: list[np.ndarray] = []
    benchmark_records: list[dict] = []
    evaluation_start = time.perf_counter()
    evaluated_count = 0
    reused_count = 0
    trained_count = 0
    failed_count = 0
    resolved_worker_count = 1
    pending_tasks: list[ParallelGenomeTask] = []
    pending_meta: list[tuple[int, Genome, int, float, int]] = []
    parallel_batch = (
        ParallelBatchContext(
            benchmark_name=benchmark_spec.name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            num_classes=num_classes,
            task=benchmark_spec.task,
            layer_norm=config.training.layer_norm,
            lr_schedule=config.training.lr_schedule,
            weight_decay=config.training.weight_decay,
            grad_clip_norm=config.training.grad_clip_norm or 1.0,
        )
        if parallel_eval is not None and parallel_eval.enabled()
        else None
    )

    for genome_idx, genome in enumerate(state.population):
        plan = _make_evaluation_plan(
            genome=genome,
            config=config,
            input_dim=input_dim,
            num_classes=num_classes,
            task=benchmark_spec.task,
            cache=cache,
            cache_namespace=benchmark_spec.name,
            multi_fidelity_schedule=multi_fidelity_schedule,
            generation=state.generation,
            evaluation_memo=evaluation_memo,
        )
        if plan["reused"]:
            result = plan["result"]
            mb = plan["model_bytes"]
            fitness = result.native_fitness
            reused_count += 1
        elif parallel_eval is not None and parallel_eval.enabled():
            pending_tasks.append(
                ParallelGenomeTask(
                    genome_dict=copy.deepcopy(genome_to_light_dict(genome)),
                    epochs=plan["epochs"],
                    lr=plan["lr"],
                    batch_size=plan["batch_size"],
                    weight_snapshot=plan["weight_snapshot"],
                )
            )
            pending_meta.append((genome_idx, genome, plan["epochs"], plan["lr"], plan["batch_size"]))
            continue
        else:
            try:
                result, mb = _evaluate_single(
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
                    cache_namespace=benchmark_spec.name,
                    multi_fidelity_schedule=multi_fidelity_schedule,
                    generation=state.generation,
                    evaluation_memo=evaluation_memo,
                )
                fitness = result.native_fitness
                evaluated_count += 0 if getattr(genome, "_last_eval_reused", False) else 1
                trained_count += 1
                if result.failure_reason is not None:
                    failed_count += 1
            except (ValueError, KeyError, RuntimeError):
                result = _failure_result(benchmark_spec.task, "runtime_error")
                fitness = float("inf")
                mb = estimate_model_bytes(genome)
                evaluated_count += 1
                trained_count += 1
                failed_count += 1

        fitnesses.append(fitness)
        model_bytes_list.append(mb)
        behaviors.append(compute_behavior(genome))
        benchmark_records.append(
            _benchmark_record(
                benchmark_name=benchmark_spec.name,
                genome=genome,
                genome_idx=genome_idx,
                result=result,
            )
        )

    if pending_tasks:
        assert parallel_batch is not None
        for (genome_idx, genome, epochs, lr, batch_size), completed in zip(
            pending_meta,
            parallel_eval.evaluate_genomes(parallel_batch, pending_tasks),
            strict=True,
        ):
            resolved_worker_count = parallel_eval.last_resolved_workers
            result = completed.result
            mb = completed.model_bytes
            _apply_completed_result(
                genome=genome,
                result=result,
                model_bytes=mb,
                cache=cache,
                cache_namespace=benchmark_spec.name,
                weights=completed.weights,
                evaluation_memo=evaluation_memo,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
            evaluated_count += 1
            trained_count += 1
            if result.failure_reason is not None:
                failed_count += 1
            fitnesses.insert(genome_idx, result.native_fitness)
            model_bytes_list.insert(genome_idx, mb)
            behaviors.insert(genome_idx, compute_behavior(genome))
            benchmark_records.insert(
                genome_idx,
                _benchmark_record(
                    benchmark_name=benchmark_spec.name,
                    genome=genome,
                    genome_idx=genome_idx,
                    result=result,
                ),
            )

    evaluation_seconds = time.perf_counter() - evaluation_start
    timing_record = _benchmark_timing_record(
        benchmark_name=benchmark_spec.name,
        benchmark_order=0,
        benchmark_total=1,
        task=benchmark_spec.task,
        data_load_seconds=data_load_seconds,
        evaluation_seconds=evaluation_seconds,
        reused_count=reused_count,
        trained_count=trained_count,
        failed_count=failed_count,
        requested_worker_count=(
            parallel_eval.last_requested_workers if parallel_eval is not None else 1
        ),
        resolved_worker_count=resolved_worker_count,
    )

    state.fitnesses = fitnesses
    state.model_bytes = model_bytes_list
    state.behaviors = behaviors
    state.benchmark_results = [_best_record(benchmark_records)]
    state.benchmark_timings = [timing_record]
    state.raw_losses = {benchmark_spec.name: fitnesses}
    state.total_evaluations += evaluated_count
    state.cache_reused = reused_count
    state.cache_trained = trained_count
    state.cache_failed = failed_count
    if progress_callback is not None:
        progress_callback("complete", timing_record)
    return state


def _prepare_benchmark_data(
    benchmark_spec: BenchmarkSpec,
    *,
    seed: int,
    validation_split: float,
    data_cache: BenchmarkDataCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load benchmark data and apply preprocessing when appropriate."""
    if data_cache is not None:
        return data_cache.get(
            benchmark_spec,
            seed=seed,
            validation_split=validation_split,
        )

    X_train, y_train, X_val, y_val = benchmark_spec.load_data(
        seed=seed, validation_split=validation_split,
    )

    if benchmark_spec.task == "language_modeling":
        return (
            X_train.astype(np.int32, copy=False),
            y_train.astype(np.int64, copy=False),
            X_val.astype(np.int32, copy=False),
            y_val.astype(np.int64, copy=False),
        )

    pp = Preprocessor()
    X_train = pp.fit_transform(X_train)
    X_val = pp.transform(X_val)
    return X_train, y_train, X_val, y_val


def _resolve_model_output_dim(
    *,
    benchmark_spec: BenchmarkSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> int:
    """Resolve output dimension, expanding LM vocab if cached tokens exceed catalog cap."""
    declared = benchmark_spec.num_classes or 1
    if benchmark_spec.task != "language_modeling":
        return declared
    observed_max = max(
        int(np.max(X_train)) if X_train.size else 0,
        int(np.max(y_train)) if y_train.size else 0,
    )
    return max(declared, observed_max + 1)


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
    cache_namespace: str = "",
    evaluation_memo: EvaluationMemo | None = None,
) -> tuple[EvaluationResult, int]:
    """Compile, optionally load cached weights, train, return (result, model_bytes)."""
    tc = config.training

    plan = _make_evaluation_plan(
        genome=genome,
        config=config,
        input_dim=input_dim,
        num_classes=num_classes,
        task=task,
        cache=cache,
        cache_namespace=cache_namespace,
        multi_fidelity_schedule=multi_fidelity_schedule,
        generation=generation,
        evaluation_memo=evaluation_memo,
    )
    if plan["reused"]:
        return plan["result"], plan["model_bytes"]
    epochs = int(plan["epochs"])
    lr = float(plan["lr"])
    batch_size = int(plan["batch_size"])
    weight_snapshot = plan["weight_snapshot"]

    model = compile_genome(
        genome,
        input_dim=input_dim,
        num_classes=num_classes,
        task=task,
        layer_norm=tc.layer_norm,
    )
    if weight_snapshot is not None:
        load_weight_snapshot(model, weight_snapshot)

    # Train
    result = train_and_evaluate(
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
    weights = extract_weights(model) if result.native_fitness != float("inf") else None

    # Model size
    mb = effective_model_bytes(genome, input_dim=input_dim, num_classes=num_classes)
    _apply_completed_result(
        genome=genome,
        result=result,
        model_bytes=mb,
        cache=cache,
        cache_namespace=cache_namespace,
        weights=weights,
        evaluation_memo=evaluation_memo,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    return result, mb


def evaluate_pool(
    state: GenerationState,
    config: RunConfig,
    benchmark_specs: list[BenchmarkSpec],
    cache: WeightCache | None = None,
    data_cache: BenchmarkDataCache | None = None,
    evaluation_memo: EvaluationMemo | None = None,
    parallel_eval: ParallelEvaluator | None = None,
    progress_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> GenerationState:
    """Multi-benchmark evaluation over the supplied benchmark sample."""
    pool_cfg = config.benchmark_pool
    if pool_cfg is None or not benchmark_specs:
        return state

    tc = config.training
    raw_losses: dict[str, list[float]] = {}
    best_records: list[dict] = []
    benchmark_timings: list[dict] = []
    evaluated_count = 0
    cache_reused_total = 0
    cache_trained_total = 0
    cache_failed_total = 0

    for benchmark_order, spec in enumerate(benchmark_specs):
        if progress_callback is not None:
            progress_callback(
                "start",
                {
                    "benchmark_name": spec.name,
                    "benchmark_order": benchmark_order,
                    "benchmark_total": len(benchmark_specs),
                    "task": spec.task,
                },
            )
        load_start = time.perf_counter()
        X_train, y_train, X_val, y_val = _prepare_benchmark_data(
            spec,
            seed=config.seed,
            validation_split=tc.validation_split,
            data_cache=data_cache,
        )
        data_load_seconds = time.perf_counter() - load_start

        input_dim = spec.input_dim or X_train.shape[1]
        num_classes = _resolve_model_output_dim(
            benchmark_spec=spec,
            X_train=X_train,
            y_train=y_train,
        )

        losses: list[float] = []
        benchmark_records: list[dict] = []
        evaluation_start = time.perf_counter()
        pending_tasks: list[ParallelGenomeTask] = []
        pending_meta: list[tuple[int, Genome, int, float, int]] = []
        reused_count = 0
        trained_count = 0
        failed_count = 0
        resolved_worker_count = 1
        parallel_batch = (
            ParallelBatchContext(
                benchmark_name=spec.name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_dim=input_dim,
                num_classes=num_classes,
                task=spec.task,
                layer_norm=config.training.layer_norm,
                lr_schedule=config.training.lr_schedule,
                weight_decay=config.training.weight_decay,
                grad_clip_norm=config.training.grad_clip_norm or 1.0,
            )
            if parallel_eval is not None and parallel_eval.enabled()
            else None
        )
        for genome_idx, genome in enumerate(state.population):
            plan = _make_evaluation_plan(
                genome=genome,
                config=config,
                input_dim=input_dim,
                num_classes=num_classes,
                task=spec.task,
                cache=cache,
                cache_namespace=spec.name,
                multi_fidelity_schedule=config.training.multi_fidelity_schedule,
                generation=state.generation,
                evaluation_memo=evaluation_memo,
            )
            if plan["reused"]:
                result = plan["result"]
                fitness = result.native_fitness
                reused_count += 1
                losses.append(fitness)
                benchmark_records.append(
                    _benchmark_record(
                        benchmark_name=spec.name,
                        genome=genome,
                        genome_idx=genome_idx,
                        result=result,
                    )
                )
                continue
            if parallel_eval is not None and parallel_eval.enabled():
                pending_tasks.append(
                    ParallelGenomeTask(
                        genome_dict=copy.deepcopy(genome_to_light_dict(genome)),
                        epochs=plan["epochs"],
                        lr=plan["lr"],
                        batch_size=plan["batch_size"],
                        weight_snapshot=plan["weight_snapshot"],
                    )
                )
                pending_meta.append((genome_idx, genome, plan["epochs"], plan["lr"], plan["batch_size"]))
                losses.append(float("nan"))
                benchmark_records.append({})
                continue
            try:
                result, _ = _evaluate_single(
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
                    cache_namespace=spec.name,
                    multi_fidelity_schedule=config.training.multi_fidelity_schedule,
                    generation=state.generation,
                    evaluation_memo=evaluation_memo,
                )
                fitness = result.native_fitness
                evaluated_count += 0 if getattr(genome, "_last_eval_reused", False) else 1
                trained_count += 1
                if result.failure_reason is not None:
                    failed_count += 1
            except (ValueError, KeyError, RuntimeError):
                result = _failure_result(spec.task, "runtime_error")
                fitness = float("inf")
                evaluated_count += 1
                trained_count += 1
                failed_count += 1
            losses.append(fitness)
            benchmark_records.append(
                _benchmark_record(
                    benchmark_name=spec.name,
                    genome=genome,
                    genome_idx=genome_idx,
                    result=result,
                )
            )

        if pending_tasks:
            assert parallel_batch is not None
            for (genome_idx, genome, epochs, lr, batch_size), completed in zip(
                pending_meta,
                parallel_eval.evaluate_genomes(parallel_batch, pending_tasks),
                strict=True,
            ):
                resolved_worker_count = parallel_eval.last_resolved_workers
                result = completed.result
                _apply_completed_result(
                    genome=genome,
                    result=result,
                    model_bytes=completed.model_bytes,
                    cache=cache,
                    cache_namespace=spec.name,
                    weights=completed.weights,
                    evaluation_memo=evaluation_memo,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                )
                losses[genome_idx] = result.native_fitness
                benchmark_records[genome_idx] = _benchmark_record(
                    benchmark_name=spec.name,
                    genome=genome,
                    genome_idx=genome_idx,
                    result=result,
                )
                evaluated_count += 1
                trained_count += 1
                if result.failure_reason is not None:
                    failed_count += 1

        raw_losses[spec.name] = losses
        best_records.append(_best_record(benchmark_records))
        timing_record = _benchmark_timing_record(
            benchmark_name=spec.name,
            benchmark_order=benchmark_order,
            benchmark_total=len(benchmark_specs),
            task=spec.task,
            data_load_seconds=data_load_seconds,
            evaluation_seconds=time.perf_counter() - evaluation_start,
            reused_count=reused_count,
            trained_count=trained_count,
            failed_count=failed_count,
            requested_worker_count=(
                parallel_eval.last_requested_workers if parallel_eval is not None else 1
            ),
            resolved_worker_count=resolved_worker_count,
        )
        benchmark_timings.append(timing_record)
        cache_reused_total += reused_count
        cache_trained_total += trained_count
        cache_failed_total += failed_count
        if progress_callback is not None:
            progress_callback("complete", timing_record)

    # Aggregate via percentile fitness
    state.fitnesses = compute_percentile_fitness(raw_losses)
    state.model_bytes = [
        estimate_model_bytes(g) for g in state.population
    ]
    state.behaviors = [compute_behavior(g) for g in state.population]
    state.benchmark_results = best_records
    state.benchmark_timings = benchmark_timings
    state.raw_losses = raw_losses
    state.total_evaluations += evaluated_count
    state.cache_reused = cache_reused_total
    state.cache_trained = cache_trained_total
    state.cache_failed = cache_failed_total
    return state


def genome_to_light_dict(genome: Genome) -> dict:
    d = {
        "layers": [g.model_dump(mode="json") for g in genome.layers],
        "connections": [g.model_dump(mode="json") for g in genome.connections],
    }
    if genome.learning_rate is not None:
        d["learning_rate"] = genome.learning_rate
    if genome.batch_size is not None:
        d["batch_size"] = genome.batch_size
    return d


def _make_evaluation_plan(
    *,
    genome: Genome,
    config: RunConfig,
    input_dim: int,
    num_classes: int,
    task: str,
    cache: WeightCache | None,
    cache_namespace: str,
    multi_fidelity_schedule: list[float] | None,
    generation: int,
    evaluation_memo: EvaluationMemo | None,
) -> dict[str, object]:
    tc = config.training
    genome_lr = getattr(genome, "learning_rate", None)
    genome_batch_size = getattr(genome, "batch_size", None)
    lr = genome_lr if genome_lr is not None else tc.learning_rate
    batch_size = genome_batch_size if genome_batch_size is not None else tc.batch_size

    weight_snapshot = None
    base_epochs = tc.epochs
    cache_lookup = getattr(cache, "lookup", None) if cache is not None else None
    cache_lookup_partial = getattr(cache, "lookup_partial", None) if cache is not None else None
    if callable(cache_lookup):
        exact = cache_lookup(genome, namespace=cache_namespace)
        if exact is not None:
            weight_snapshot = exact
            base_epochs = max(1, int(base_epochs * tc.finetune_epoch_ratio))
        elif callable(cache_lookup_partial):
            partial = cache_lookup_partial(genome, namespace=cache_namespace)
            if partial is not None:
                weight_snapshot = partial
                base_epochs = max(1, int(base_epochs * tc.partial_epoch_ratio))

    epochs = _apply_epoch_scaling(
        config=config,
        generation=generation,
        base_epochs=base_epochs,
        multi_fidelity_schedule=multi_fidelity_schedule,
    )

    cached_runtime = _lookup_runtime_result(genome, cache_namespace)
    if cached_runtime is not None:
        result, mb = cached_runtime
        genome._last_eval_reused = True
        genome.fitness = result.native_fitness
        genome.model_bytes = mb
        return {"reused": True, "result": result, "model_bytes": mb}

    if evaluation_memo is not None:
        cached_eval = evaluation_memo.lookup(
            benchmark_name=cache_namespace,
            genome=genome,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        if cached_eval is not None:
            result, mb = cached_eval
            _store_runtime_result(genome, cache_namespace, result, mb)
            genome._last_eval_reused = True
            genome.fitness = result.native_fitness
            genome.model_bytes = mb
            return {"reused": True, "result": result, "model_bytes": mb}

    return {
        "reused": False,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "weight_snapshot": weight_snapshot,
    }


def _apply_completed_result(
    *,
    genome: Genome,
    result: EvaluationResult,
    model_bytes: int,
    cache: WeightCache | None,
    cache_namespace: str,
    weights: dict[str, np.ndarray] | None,
    evaluation_memo: EvaluationMemo | None,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    genome._last_eval_reused = False
    genome.fitness = result.native_fitness
    genome.model_bytes = model_bytes
    _store_runtime_result(genome, cache_namespace, result, model_bytes)
    if cache is not None and weights is not None and result.native_fitness != float("inf"):
        cache.store(genome, weights, namespace=cache_namespace)
    if evaluation_memo is not None:
        evaluation_memo.store(
            benchmark_name=cache_namespace,
            genome=genome,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            result=result,
            model_bytes=model_bytes,
        )


def _apply_epoch_scaling(
    *,
    config: RunConfig,
    generation: int,
    base_epochs: int,
    multi_fidelity_schedule: list[float] | None,
) -> int:
    tc = config.training
    epochs = base_epochs
    if multi_fidelity_schedule is not None and generation < len(multi_fidelity_schedule):
        scale = multi_fidelity_schedule[generation]
        epochs = max(1, int(epochs * scale))
    elif tc.multi_fidelity and multi_fidelity_schedule is None:
        total_gens = config.evolution.num_generations
        if total_gens > 1:
            progress = generation / (total_gens - 1)
            scale = 0.3 + 0.7 * progress
            epochs = max(1, int(epochs * scale))
    return epochs


def _lookup_runtime_result(
    genome: Genome,
    benchmark_name: str,
) -> tuple[EvaluationResult, int] | None:
    cache = getattr(genome, "_eval_cache", None)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(benchmark_name)
    if not isinstance(entry, tuple) or len(entry) != 2:
        return None
    result, model_bytes = entry
    return copy.deepcopy(result), int(model_bytes)


def _store_runtime_result(
    genome: Genome,
    benchmark_name: str,
    result: EvaluationResult,
    model_bytes: int,
) -> None:
    cache = getattr(genome, "_eval_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(genome, "_eval_cache", cache)
    cache[benchmark_name] = (copy.deepcopy(result), int(model_bytes))


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


def _failure_result(task: str, reason: str) -> EvaluationResult:
    metric_name = "perplexity" if task == "language_modeling" else ("mse" if task == "regression" else "accuracy")
    metric_direction = "min" if task in {"regression", "language_modeling"} else "max"
    return EvaluationResult(
        metric_name=metric_name,
        metric_direction=metric_direction,
        metric_value=float("nan"),
        quality=float("-inf"),
        native_fitness=float("inf"),
        train_seconds=0.0,
        failure_reason=reason,
    )


def _benchmark_record(
    *,
    benchmark_name: str,
    genome: Genome,
    genome_idx: int,
    result,
) -> dict:
    metric_value = None if math.isnan(result.metric_value) else float(result.metric_value)
    quality = None if math.isinf(result.quality) else float(result.quality)
    return {
        "benchmark_name": benchmark_name,
        "metric_name": result.metric_name,
        "metric_direction": result.metric_direction,
        "metric_value": metric_value,
        "quality": quality,
        "parameter_count": genome.param_count,
        "train_seconds": float(result.train_seconds),
        "architecture_summary": f"{len(genome.enabled_layers)}L/{len(genome.enabled_connections)}C",
        "genome_id": None,
        "genome_idx": genome_idx,
        "status": "ok" if result.failure_reason is None else "failed",
        "failure_reason": result.failure_reason,
    }


def _best_record(records: list[dict]) -> dict:
    if not records:
        raise ValueError("expected benchmark records")
    return min(records, key=_record_sort_key)


def _record_sort_key(record: dict) -> tuple:
    status_rank = 0 if record["status"] == "ok" else 1
    metric_value = record["metric_value"]
    if metric_value is None:
        metric_rank = float("inf")
    elif record["metric_direction"] == "min":
        metric_rank = float(metric_value)
    else:
        metric_rank = -float(metric_value)
    return (status_rank, metric_rank, record["genome_idx"])


def _benchmark_timing_record(
    *,
    benchmark_name: str,
    benchmark_order: int,
    benchmark_total: int,
    task: str,
    data_load_seconds: float,
    evaluation_seconds: float,
    reused_count: int,
    trained_count: int,
    failed_count: int,
    requested_worker_count: int,
    resolved_worker_count: int,
) -> dict[str, object]:
    return {
        "benchmark_name": benchmark_name,
        "benchmark_order": benchmark_order,
        "benchmark_total": benchmark_total,
        "task": task,
        "data_load_seconds": round(float(data_load_seconds), 6),
        "evaluation_seconds": round(float(evaluation_seconds), 6),
        "total_seconds": round(float(data_load_seconds + evaluation_seconds), 6),
        "reused_count": int(reused_count),
        "trained_count": int(trained_count),
        "failed_count": int(failed_count),
        "requested_worker_count": int(requested_worker_count),
        "resolved_worker_count": int(resolved_worker_count),
    }
