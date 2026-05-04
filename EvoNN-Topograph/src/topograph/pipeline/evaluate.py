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
from topograph.runtime.backends import resolve_runtime_backend_with_policy


@dataclass
class GenerationState:
    generation: int
    population: list[Genome]
    fitnesses: list[float] = field(default_factory=list)
    model_bytes: list[int] = field(default_factory=list)
    behaviors: list[np.ndarray] = field(default_factory=list)
    benchmark_results: list[dict] = field(default_factory=list)
    benchmark_timings: list[dict] = field(default_factory=list)
    benchmark_families: dict[str, str] = field(default_factory=dict)
    raw_losses: dict[str, list[float]] = field(default_factory=dict)
    phase: str = "explore"
    total_evaluations: int = 0
    cache_reused: int = 0
    cache_trained: int = 0
    cache_failed: int = 0
    active_benchmark_family: str | None = None


class BenchmarkDataCache:
    """In-memory cache of loaded and preprocessed benchmark arrays."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, float], tuple[np.ndarray, ...]] = {}
        self._hits: dict[str, int] = {}
        self._misses: dict[str, int] = {}

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
            self._hits[benchmark_spec.name] = self._hits.get(benchmark_spec.name, 0) + 1
            return cached
        self._misses[benchmark_spec.name] = self._misses.get(benchmark_spec.name, 0) + 1

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

    def consume_stats(self, benchmark_name: str) -> tuple[int, int]:
        hits = self._hits.pop(benchmark_name, 0)
        misses = self._misses.pop(benchmark_name, 0)
        return hits, misses


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
    data_cache_hits, data_cache_misses = _consume_data_cache_stats(data_cache, benchmark_spec.name)

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
    worker_clamp_reason = "sequential"
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
    benchmark_family = benchmark_family_name(benchmark_spec)
    family_namespace = (
        _family_cache_namespace(benchmark_family)
        if _family_transfer_enabled(config)
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
            family_namespace=family_namespace,
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
                    family_namespace=family_namespace,
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
            worker_clamp_reason = parallel_eval.last_clamp_reason
            result = completed.result
            mb = completed.model_bytes
            _apply_completed_result(
                genome=genome,
                result=result,
                model_bytes=mb,
                cache=cache,
                cache_namespace=benchmark_spec.name,
                family_namespace=family_namespace,
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
        data_cache_hits=data_cache_hits,
        data_cache_misses=data_cache_misses,
        worker_clamp_reason=worker_clamp_reason,
    )

    state.fitnesses = fitnesses
    state.model_bytes = model_bytes_list
    state.behaviors = behaviors
    state.benchmark_results = [_best_record(benchmark_records)]
    state.benchmark_timings = [timing_record]
    state.benchmark_families = {benchmark_spec.name: benchmark_family}
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


def _consume_data_cache_stats(
    data_cache: BenchmarkDataCache | None,
    benchmark_name: str,
) -> tuple[int, int]:
    if data_cache is None:
        return (0, 0)
    return data_cache.consume_stats(benchmark_name)


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


def benchmark_family_name(benchmark_spec: BenchmarkSpec) -> str:
    task = getattr(benchmark_spec, "task", "")
    source = getattr(benchmark_spec, "source", "")
    if task == "language_modeling":
        return "language_modeling"
    if source == "image":
        return "image"
    return "tabular"


def _family_cache_namespace(family: str) -> str:
    return f"family::{family}"


def _family_transfer_enabled(config: RunConfig) -> bool:
    pool_cfg = config.benchmark_pool
    if pool_cfg is None:
        return False
    if isinstance(pool_cfg, dict):
        return bool(pool_cfg.get("family_transfer", False))
    return bool(pool_cfg.family_transfer)


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
    family_namespace: str | None = None,
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
        family_namespace=family_namespace,
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
    runtime_selection = resolve_runtime_backend_with_policy(
        config.runtime.backend,
        allow_fallback=config.runtime.allow_fallback,
    )
    if runtime_selection.resolved_backend == "numpy-fallback":
        result, mb = _evaluate_single_numpy_fallback(
            genome=genome,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            num_classes=num_classes,
            task=task,
        )
        _apply_completed_result(
            genome=genome,
            result=result,
            model_bytes=mb,
            cache=cache,
            cache_namespace=cache_namespace,
            family_namespace=family_namespace,
            weights=None,
            evaluation_memo=evaluation_memo,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        return result, mb

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
        family_namespace=family_namespace,
        weights=weights,
        evaluation_memo=evaluation_memo,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    return result, mb


def _evaluate_single_numpy_fallback(
    *,
    genome: Genome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    num_classes: int,
    task: str,
) -> tuple[EvaluationResult, int]:
    started = time.perf_counter()
    try:
        if task == "regression":
            predictions = _fallback_regression(X_train, y_train, X_val)
        elif task == "language_modeling":
            predictions = _fallback_language_probs(y_train, y_val, num_classes)
        else:
            predictions = _fallback_classification(X_train, y_train, X_val)
        metric_name, metric_direction, metric_value, quality = _compute_fallback_metric(task, y_val, predictions)
        native_fitness = -quality if metric_direction == "max" else metric_value
        result = EvaluationResult(
            metric_name=metric_name,
            metric_direction=metric_direction,
            metric_value=metric_value,
            quality=quality,
            native_fitness=float(native_fitness),
            train_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        result = _failure_result(task, f"numpy_fallback_error:{type(exc).__name__}")
    return result, effective_model_bytes(genome, input_dim=input_dim, num_classes=num_classes)


def _fallback_classification(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    x_train, x_val = _standardize_pair(X_train, X_val)
    classes = np.unique(y_train.astype(int, copy=False))
    centroids = []
    for label in classes:
        mask = y_train == label
        centroids.append(x_train[mask].mean(axis=0) if np.any(mask) else np.zeros(x_train.shape[1], dtype=np.float32))
    centroid_matrix = np.asarray(centroids, dtype=np.float32)
    distances = np.sum((x_val[:, None, :] - centroid_matrix[None, :, :]) ** 2, axis=2)
    return classes[np.argmin(distances, axis=1)]


def _fallback_regression(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    x_train, x_val = _standardize_pair(X_train, X_val)
    design = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    ridge = np.eye(design.shape[1], dtype=np.float32) * 1e-4
    ridge[-1, -1] = 0.0
    coeffs = np.linalg.solve(design.T @ design + ridge, design.T @ y_train.astype(np.float32))
    val_design = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=np.float32)], axis=1)
    return val_design @ coeffs


def _fallback_language_probs(y_train: np.ndarray, y_val: np.ndarray, num_classes: int) -> np.ndarray:
    targets = y_train.reshape(-1).astype(int)
    vocab_size = max(int(num_classes), int(targets.max(initial=0)) + 1, int(y_val.max(initial=0)) + 1)
    counts = np.bincount(targets, minlength=vocab_size).astype(np.float32) + 1.0
    probs = counts / counts.sum()
    if y_val.ndim == 2:
        return np.broadcast_to(probs, (*y_val.shape, vocab_size)).copy()
    return np.broadcast_to(probs, (len(y_val), vocab_size)).copy()


def _standardize_pair(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=np.float32).reshape(X_train.shape[0], -1)
    val = np.asarray(X_val, dtype=np.float32).reshape(X_val.shape[0], -1)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (val - mean) / std


def _compute_fallback_metric(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[str, str, float, float]:
    if task == "classification":
        accuracy = float(np.mean(y_pred.ravel().astype(int) == y_true.ravel().astype(int)))
        return "accuracy", "max", accuracy, accuracy
    if task == "language_modeling":
        probs = np.clip(y_pred, 1e-8, 1.0)
        targets = y_true.reshape(-1).astype(int)
        flat = probs.reshape(-1, probs.shape[-1])
        cross_entropy = float(-np.mean(np.log(flat[np.arange(len(targets)), targets])))
        perplexity = float(np.exp(np.clip(cross_entropy, -20.0, 20.0)))
        return "perplexity", "min", perplexity, -perplexity
    mse = float(np.mean((y_pred.ravel() - y_true.ravel()) ** 2))
    return "mse", "min", mse, -mse


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
    benchmark_families: dict[str, str] = {}
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
        data_cache_hits, data_cache_misses = _consume_data_cache_stats(data_cache, spec.name)

        input_dim = spec.input_dim or X_train.shape[1]
        benchmark_family = benchmark_family_name(spec)
        benchmark_families[spec.name] = benchmark_family
        family_namespace = (
            _family_cache_namespace(benchmark_family)
            if _family_transfer_enabled(config)
            else None
        )
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
        worker_clamp_reason = "sequential"
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
                family_namespace=family_namespace,
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
                    family_namespace=family_namespace,
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
                worker_clamp_reason = parallel_eval.last_clamp_reason
                result = completed.result
                _apply_completed_result(
                    genome=genome,
                    result=result,
                    model_bytes=completed.model_bytes,
                    cache=cache,
                    cache_namespace=spec.name,
                    family_namespace=family_namespace,
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
            data_cache_hits=data_cache_hits,
            data_cache_misses=data_cache_misses,
            worker_clamp_reason=worker_clamp_reason,
        )
        benchmark_timings.append(timing_record)
        cache_reused_total += reused_count
        cache_trained_total += trained_count
        cache_failed_total += failed_count
        if progress_callback is not None:
            progress_callback("complete", timing_record)

    state.fitnesses = _aggregate_pool_fitness(
        raw_losses=raw_losses,
        benchmark_families=benchmark_families,
        benchmark_timings=benchmark_timings,
        aggregation=pool_cfg.aggregation,
        active_family=state.active_benchmark_family,
        family_focus_weight=pool_cfg.family_focus_weight,
        benchmark_cost_penalty_alpha=pool_cfg.benchmark_cost_penalty_alpha,
    )
    state.model_bytes = [
        estimate_model_bytes(g) for g in state.population
    ]
    state.behaviors = [compute_behavior(g) for g in state.population]
    state.benchmark_results = best_records
    state.benchmark_timings = benchmark_timings
    state.benchmark_families = benchmark_families
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
    family_namespace: str | None,
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
        if weight_snapshot is None and family_namespace:
            exact = cache_lookup(genome, namespace=family_namespace)
            if exact is not None:
                weight_snapshot = exact
                base_epochs = max(1, int(base_epochs * tc.finetune_epoch_ratio))
            elif callable(cache_lookup_partial):
                partial = cache_lookup_partial(genome, namespace=family_namespace)
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
    family_namespace: str | None,
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
        if family_namespace:
            cache.store(genome, weights, namespace=family_namespace)
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


def _aggregate_pool_fitness(
    *,
    raw_losses: dict[str, list[float]],
    benchmark_families: dict[str, str],
    benchmark_timings: list[dict[str, object]],
    aggregation: str,
    active_family: str | None,
    family_focus_weight: float,
    benchmark_cost_penalty_alpha: float,
) -> list[float]:
    if aggregation == "percentile" or not raw_losses:
        return compute_percentile_fitness(raw_losses)

    family_scores: dict[str, list[float]] = {}
    for family in sorted(set(benchmark_families.get(name, "tabular") for name in raw_losses)):
        family_losses = {
            name: losses
            for name, losses in raw_losses.items()
            if benchmark_families.get(name, "tabular") == family
        }
        family_scores[family] = _family_weighted_percentile_fitness(
            family_losses=family_losses,
            benchmark_timings=benchmark_timings,
            alpha=benchmark_cost_penalty_alpha,
        )

    if not family_scores:
        return compute_percentile_fitness(raw_losses)

    ordered_families = list(sorted(family_scores))
    numerators = [0.0] * len(next(iter(family_scores.values())))
    denominators = [0.0] * len(numerators)
    for family in ordered_families:
        weight = family_focus_weight if active_family and family == active_family else 1.0
        for idx, score in enumerate(family_scores[family]):
            numerators[idx] += weight * score
            denominators[idx] += weight
    return [numerators[idx] / max(denominators[idx], 1e-12) for idx in range(len(numerators))]


def _family_weighted_percentile_fitness(
    *,
    family_losses: dict[str, list[float]],
    benchmark_timings: list[dict[str, object]],
    alpha: float,
) -> list[float]:
    if not family_losses:
        return []

    per_benchmark_percentiles: dict[str, list[float]] = {}
    for benchmark_name, losses in family_losses.items():
        per_benchmark_percentiles[benchmark_name] = compute_percentile_fitness(
            {benchmark_name: losses}
        )

    timing_lookup = {
        str(row["benchmark_name"]): float(row["evaluation_seconds"])
        for row in benchmark_timings
        if str(row["benchmark_name"]) in family_losses
    }
    costs = [timing_lookup.get(name, 1.0) for name in family_losses]
    median_cost = _median_value(costs) if costs else 1.0

    sample_len = len(next(iter(per_benchmark_percentiles.values())))
    numerators = [0.0] * sample_len
    denominators = [0.0] * sample_len
    for benchmark_name, percentiles in per_benchmark_percentiles.items():
        cost = timing_lookup.get(benchmark_name, median_cost)
        ratio = max(cost / max(median_cost, 1e-6), 1e-6)
        weight = ratio ** (-alpha) if alpha > 0.0 else 1.0
        for idx, score in enumerate(percentiles):
            numerators[idx] += weight * score
            denominators[idx] += weight
    return [numerators[idx] / max(denominators[idx], 1e-12) for idx in range(sample_len)]


def _median_value(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


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
    data_cache_hits: int,
    data_cache_misses: int,
    worker_clamp_reason: str,
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
        "data_cache_hits": int(data_cache_hits),
        "data_cache_misses": int(data_cache_misses),
        "worker_clamp_reason": str(worker_clamp_reason),
    }
