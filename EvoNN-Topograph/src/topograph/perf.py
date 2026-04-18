"""Performance regression guardrails for completed Topograph runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topograph.export.report import _resolve_run_id
from topograph.storage import RunStore


class PerformanceRegressionError(AssertionError):
    """Raised when a completed run violates performance guardrails."""


def performance_summary(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    store = RunStore(run_path / "metrics.duckdb")
    try:
        run_id = _resolve_run_id(store)
        budget = store.load_budget_metadata(run_id) or {}
        timings = store.load_benchmark_timings(run_id)
    finally:
        store.close()

    return {
        "run_dir": str(run_path),
        "wall_clock_seconds": budget.get("wall_clock_seconds"),
        "evaluation_count": budget.get("evaluation_count"),
        "cache_reuse_rate": budget.get("cache_reuse_rate"),
        "resolved_parallel_workers_max": budget.get("resolved_parallel_workers_max"),
        "benchmark_total_seconds": budget.get("benchmark_total_seconds"),
        "worker_clamp_reason_counts": budget.get("worker_clamp_reason_counts", {}),
        "data_cache_hits": budget.get("data_cache_hits", 0),
        "data_cache_misses": budget.get("data_cache_misses", 0),
        "benchmark_rows": len(timings),
    }


def assert_performance_guardrails(
    run_dir: str | Path,
    *,
    max_wall_clock_seconds: float | None = None,
    max_resolved_workers: int | None = None,
    min_cache_reuse_rate: float | None = None,
    max_benchmark_total_seconds: float | None = None,
) -> dict[str, Any]:
    summary = performance_summary(run_dir)
    violations: list[str] = []

    wall_clock = summary.get("wall_clock_seconds")
    if max_wall_clock_seconds is not None and wall_clock is not None:
        if float(wall_clock) > float(max_wall_clock_seconds):
            violations.append(
                f"wall_clock_seconds {float(wall_clock):.3f} > {float(max_wall_clock_seconds):.3f}"
            )

    resolved_workers = summary.get("resolved_parallel_workers_max")
    if max_resolved_workers is not None and resolved_workers is not None:
        if int(resolved_workers) > int(max_resolved_workers):
            violations.append(
                f"resolved_parallel_workers_max {int(resolved_workers)} > {int(max_resolved_workers)}"
            )

    cache_reuse_rate = summary.get("cache_reuse_rate")
    if min_cache_reuse_rate is not None and cache_reuse_rate is not None:
        if float(cache_reuse_rate) < float(min_cache_reuse_rate):
            violations.append(
                f"cache_reuse_rate {float(cache_reuse_rate):.3f} < {float(min_cache_reuse_rate):.3f}"
            )

    benchmark_total = summary.get("benchmark_total_seconds")
    if max_benchmark_total_seconds is not None and benchmark_total is not None:
        if float(benchmark_total) > float(max_benchmark_total_seconds):
            violations.append(
                f"benchmark_total_seconds {float(benchmark_total):.3f} > "
                f"{float(max_benchmark_total_seconds):.3f}"
            )

    if violations:
        raise PerformanceRegressionError("; ".join(violations))

    return summary
