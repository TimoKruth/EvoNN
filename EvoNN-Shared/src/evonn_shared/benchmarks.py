"""Benchmark identity and resolution helpers shared across EvoNN packages."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


TaskKind = Literal["classification", "regression", "language_modeling"]
MetricDirection = Literal["max", "min"]


class BenchmarkDescriptor(BaseModel):
    """Minimal shared benchmark descriptor for compare-grade integration."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    task_kind: TaskKind
    metric_name: str
    metric_direction: MetricDirection
    source: str | None = None
    native_name: str | None = None


def load_benchmark_descriptors(path: str | Path) -> list[BenchmarkDescriptor]:
    """Load benchmark descriptors from a YAML list or mapping."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        benchmarks = payload.get("benchmarks", [])
    else:
        benchmarks = payload
    return [BenchmarkDescriptor.model_validate(item) for item in benchmarks]
