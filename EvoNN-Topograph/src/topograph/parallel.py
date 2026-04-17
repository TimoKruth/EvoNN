"""Parallel evaluation helpers for Topograph."""

from __future__ import annotations

import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Final

import numpy as np

from topograph.genome.codec import dict_to_genome, genome_to_dict
from topograph.nn.compiler import compile_genome
from topograph.nn.train import (
    EvaluationResult,
    effective_model_bytes,
    extract_weights,
    load_weight_snapshot,
    train_and_evaluate,
)

_AUTO_MAX_WORKERS: Final[int] = 2
_SEQUENTIAL_WORKERS: Final[int] = 1
_DEFAULT_RESERVED_SYSTEM_MEMORY_BYTES: Final[int] = 8 * 1024**3
_DEFAULT_PARALLEL_MEMORY_FRACTION: Final[float] = 0.5
_DEFAULT_PARALLEL_CPU_FRACTION: Final[float] = 0.5
_DEFAULT_WORKER_THREAD_LIMIT: Final[int] = 1
_PER_WORKER_OVERHEAD_BYTES: Final[int] = 2 * 1024**3
_DATA_MEMORY_MULTIPLIER: Final[float] = 2.5
_WEIGHT_MEMORY_MULTIPLIER: Final[float] = 1.5
_THREAD_LIMIT_ENV: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def _default_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(_AUTO_MAX_WORKERS, max(1, cpu // 4)))


def _configure_process_env_limits(thread_limit: int = _DEFAULT_WORKER_THREAD_LIMIT) -> None:
    for key, value in _THREAD_LIMIT_ENV.items():
        if key == "TOKENIZERS_PARALLELISM":
            os.environ.setdefault(key, value)
            continue
        os.environ.setdefault(key, str(thread_limit))


def _system_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    total = page_size * phys_pages
    return total if total > 0 else None


def _weight_snapshot_bytes(snapshot: dict[str, np.ndarray] | None) -> int:
    if not snapshot:
        return 0
    return sum(int(np.asarray(value).nbytes) for value in snapshot.values())


@dataclass(frozen=True)
class ParallelRuntimeLimits:
    cpu_fraction_limit: float = _DEFAULT_PARALLEL_CPU_FRACTION
    memory_fraction_limit: float = _DEFAULT_PARALLEL_MEMORY_FRACTION
    reserved_system_memory_bytes: int = _DEFAULT_RESERVED_SYSTEM_MEMORY_BYTES
    worker_thread_limit: int = _DEFAULT_WORKER_THREAD_LIMIT


@dataclass
class ParallelBatchContext:
    benchmark_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    input_dim: int
    num_classes: int
    task: str
    layer_norm: bool
    lr_schedule: str
    weight_decay: float
    grad_clip_norm: float

    @property
    def data_bytes(self) -> int:
        return (
            int(self.X_train.nbytes)
            + int(self.y_train.nbytes)
            + int(self.X_val.nbytes)
            + int(self.y_val.nbytes)
        )


@dataclass
class ParallelGenomeTask:
    genome_dict: dict
    epochs: int
    lr: float
    batch_size: int
    weight_snapshot: dict[str, np.ndarray] | None = None


@dataclass
class ParallelGenomeResult:
    genome_dict: dict
    benchmark_name: str
    result: EvaluationResult
    model_bytes: int
    weights: dict[str, np.ndarray] | None


_WORKER_BATCH_CONTEXT: ParallelBatchContext | None = None


def _init_parallel_worker(
    batch_context: ParallelBatchContext,
    runtime_limits: ParallelRuntimeLimits,
) -> None:
    global _WORKER_BATCH_CONTEXT
    _configure_process_env_limits(runtime_limits.worker_thread_limit)
    _WORKER_BATCH_CONTEXT = batch_context


def _run_parallel_genome_task_with_context(
    task: ParallelGenomeTask,
    batch_context: ParallelBatchContext,
) -> ParallelGenomeResult:
    genome = dict_to_genome(task.genome_dict)
    model = compile_genome(
        genome,
        input_dim=batch_context.input_dim,
        num_classes=batch_context.num_classes,
        task=batch_context.task,
        layer_norm=batch_context.layer_norm,
    )
    if task.weight_snapshot is not None:
        load_weight_snapshot(model, task.weight_snapshot)

    result = train_and_evaluate(
        model,
        batch_context.X_train,
        batch_context.y_train,
        batch_context.X_val,
        batch_context.y_val,
        epochs=task.epochs,
        lr=task.lr,
        batch_size=task.batch_size,
        task=batch_context.task,
        lr_schedule=batch_context.lr_schedule,
        weight_decay=batch_context.weight_decay,
        grad_clip_norm=batch_context.grad_clip_norm,
    )
    weights = extract_weights(model) if result.native_fitness != float("inf") else None
    return ParallelGenomeResult(
        genome_dict=genome_to_dict(genome),
        benchmark_name=batch_context.benchmark_name,
        result=result,
        model_bytes=effective_model_bytes(
            genome,
            input_dim=batch_context.input_dim,
            num_classes=batch_context.num_classes,
        ),
        weights=weights,
    )


def _run_parallel_genome_task(task: ParallelGenomeTask) -> ParallelGenomeResult:
    if _WORKER_BATCH_CONTEXT is None:
        raise RuntimeError("parallel worker batch context not initialized")
    return _run_parallel_genome_task_with_context(task, _WORKER_BATCH_CONTEXT)


class ParallelEvaluator:
    """Process-pool backed evaluator for expensive training tasks."""

    def __init__(
        self,
        max_workers: int = 0,
        runtime_limits: ParallelRuntimeLimits | None = None,
    ) -> None:
        self.requested_workers = max_workers or _default_workers()
        self.max_workers = self.requested_workers
        self.runtime_limits = runtime_limits or ParallelRuntimeLimits()
        self.last_requested_workers = self.requested_workers
        self.last_resolved_workers = _SEQUENTIAL_WORKERS

    def enabled(self) -> bool:
        return self.max_workers > _SEQUENTIAL_WORKERS

    def _estimate_worker_bytes(
        self,
        batch_context: ParallelBatchContext,
        tasks: list[ParallelGenomeTask],
    ) -> int:
        max_weight_bytes = max(
            (_weight_snapshot_bytes(task.weight_snapshot) for task in tasks),
            default=0,
        )
        return (
            _PER_WORKER_OVERHEAD_BYTES
            + int(batch_context.data_bytes * _DATA_MEMORY_MULTIPLIER)
            + int(max_weight_bytes * _WEIGHT_MEMORY_MULTIPLIER)
        )

    def _memory_limited_workers(
        self,
        batch_context: ParallelBatchContext,
        tasks: list[ParallelGenomeTask],
    ) -> int:
        total_memory = _system_memory_bytes()
        if total_memory is None:
            return self.max_workers

        reserved_for_system = max(
            self.runtime_limits.reserved_system_memory_bytes,
            int(total_memory * (1.0 - self.runtime_limits.memory_fraction_limit)),
        )
        parallel_budget = max(0, total_memory - reserved_for_system)
        worker_bytes = max(1, self._estimate_worker_bytes(batch_context, tasks))
        return max(1, parallel_budget // worker_bytes)

    def _resolve_worker_count(
        self,
        batch_context: ParallelBatchContext,
        tasks: list[ParallelGenomeTask],
    ) -> int:
        cpu_budget = max(
            1,
            int(math.floor((os.cpu_count() or 2) * self.runtime_limits.cpu_fraction_limit)),
        )
        memory_budget = self._memory_limited_workers(batch_context, tasks)
        return max(
            1,
            min(
                self.max_workers,
                len(tasks),
                cpu_budget,
                memory_budget,
            ),
        )

    def evaluate_genomes(
        self,
        batch_context: ParallelBatchContext,
        tasks: list[ParallelGenomeTask],
    ) -> list[ParallelGenomeResult]:
        if not tasks:
            return []

        self.last_requested_workers = self.requested_workers
        worker_count = self._resolve_worker_count(batch_context, tasks)
        self.last_resolved_workers = worker_count
        if worker_count <= _SEQUENTIAL_WORKERS or len(tasks) == 1:
            return [
                _run_parallel_genome_task_with_context(task, batch_context)
                for task in tasks
            ]

        _configure_process_env_limits(self.runtime_limits.worker_thread_limit)
        results: list[ParallelGenomeResult | None] = [None] * len(tasks)

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_parallel_worker,
            initargs=(batch_context, self.runtime_limits),
        ) as pool:
            next_task_idx = 0
            pending: dict = {}

            while next_task_idx < worker_count and next_task_idx < len(tasks):
                future = pool.submit(_run_parallel_genome_task, tasks[next_task_idx])
                pending[future] = next_task_idx
                next_task_idx += 1

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    task_idx = pending.pop(future)
                    results[task_idx] = future.result()
                    if next_task_idx < len(tasks):
                        next_future = pool.submit(_run_parallel_genome_task, tasks[next_task_idx])
                        pending[next_future] = next_task_idx
                        next_task_idx += 1

        return [result for result in results if result is not None]

    def close(self) -> None:
        return
