from __future__ import annotations

import numpy as np

import topograph.parallel as parallel_mod


def _batch() -> parallel_mod.ParallelBatchContext:
    return parallel_mod.ParallelBatchContext(
        benchmark_name="demo",
        X_train=np.zeros((8, 4), dtype=np.float32),
        y_train=np.zeros(8, dtype=np.int64),
        X_val=np.zeros((4, 4), dtype=np.float32),
        y_val=np.zeros(4, dtype=np.int64),
        input_dim=4,
        num_classes=2,
        task="classification",
        layer_norm=True,
        lr_schedule="cosine",
        weight_decay=0.01,
        grad_clip_norm=1.0,
    )


def _tasks(n: int) -> list[parallel_mod.ParallelGenomeTask]:
    return [
        parallel_mod.ParallelGenomeTask(
            genome_dict={"layers": [], "connections": []},
            epochs=1,
            lr=0.01,
            batch_size=8,
        )
        for _ in range(n)
    ]


def test_default_workers_is_conservative(monkeypatch):
    monkeypatch.setattr(parallel_mod.os, "cpu_count", lambda: 10)
    assert parallel_mod._default_workers() == 2

    monkeypatch.setattr(parallel_mod.os, "cpu_count", lambda: 4)
    assert parallel_mod._default_workers() == 1


def test_resolve_worker_count_leaves_half_cpus_free(monkeypatch):
    evaluator = parallel_mod.ParallelEvaluator(max_workers=8)
    monkeypatch.setattr(parallel_mod.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(parallel_mod, "_system_memory_bytes", lambda: 64 * 1024**3)

    workers = evaluator._resolve_worker_count(_batch(), _tasks(8))

    assert workers == 5


def test_resolve_worker_count_respects_memory_budget(monkeypatch):
    evaluator = parallel_mod.ParallelEvaluator(max_workers=8)
    monkeypatch.setattr(parallel_mod.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(parallel_mod, "_system_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(
        evaluator,
        "_estimate_worker_bytes",
        lambda batch_context, tasks: 10 * 1024**3,
    )

    workers = evaluator._resolve_worker_count(_batch(), _tasks(8))

    assert workers == 1


def test_configure_process_env_limits_preserves_existing_value(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    parallel_mod._configure_process_env_limits()

    assert parallel_mod.os.environ["OMP_NUM_THREADS"] == "3"
    assert parallel_mod.os.environ["OPENBLAS_NUM_THREADS"] == "1"


def test_runtime_limits_cap_cpu_memory_and_thread_budget(monkeypatch):
    limits = parallel_mod.ParallelRuntimeLimits(
        cpu_fraction_limit=0.25,
        memory_fraction_limit=0.25,
        reserved_system_memory_bytes=2 * 1024**3,
        worker_thread_limit=2,
    )
    evaluator = parallel_mod.ParallelEvaluator(max_workers=8, runtime_limits=limits)

    monkeypatch.setattr(parallel_mod.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(parallel_mod, "_system_memory_bytes", lambda: 32 * 1024**3)
    monkeypatch.setattr(
        evaluator,
        "_estimate_worker_bytes",
        lambda batch_context, tasks: 4 * 1024**3,
    )

    workers = evaluator._resolve_worker_count(_batch(), _tasks(8))

    assert workers == 2

    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    parallel_mod._configure_process_env_limits(limits.worker_thread_limit)
    assert parallel_mod.os.environ["OPENBLAS_NUM_THREADS"] == "2"
