"""Benchmark identity and resolution helpers shared across EvoNN packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict


TaskKind = Literal["classification", "regression", "language_modeling"]
MetricDirection = Literal["max", "min"]
BenchmarkGroup = Literal["tabular", "synthetic", "image", "language_modeling"]
BenchmarkDifficulty = Literal["smoke", "core", "hard", "stress"]
RuntimeClass = Literal["ci", "local", "overnight", "weekend", "special"]


class BenchmarkDescriptor(BaseModel):
    """Minimal shared benchmark descriptor for compare-grade integration."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    task_kind: TaskKind
    metric_name: str
    metric_direction: MetricDirection
    benchmark_group: BenchmarkGroup | None = None
    domain: str | None = None
    difficulty: BenchmarkDifficulty | None = None
    runtime_class: RuntimeClass | None = None
    minimum_required_contenders: tuple[str, ...] = ()
    enhanced_optional_contenders: tuple[str, ...] = ()
    score_ceiling: float | None = None
    tie_tolerance_abs: float = 1e-12
    tie_tolerance_rel: float = 1e-12
    admission_notes: str = ""
    source: str | None = None
    native_name: str | None = None


def _benchmark_items(payload: Any) -> Any:
    """Return the benchmark sequence from either supported YAML container shape."""

    if isinstance(payload, dict):
        return payload.get("benchmarks", [])
    return payload


def load_benchmark_descriptors(path: str | Path) -> list[BenchmarkDescriptor]:
    """Load benchmark descriptors from a YAML sequence or ``benchmarks`` mapping."""

    descriptor_path = Path(path)
    payload = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    return [BenchmarkDescriptor.model_validate(item) for item in _benchmark_items(payload)]
