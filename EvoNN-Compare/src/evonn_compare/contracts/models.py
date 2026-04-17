"""Shared export contract models for legacy and current compare systems."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


BenchmarkStatus = Literal["ok", "failed", "skipped", "unsupported", "missing"]
MetricDirection = Literal["max", "min"]
SystemName = Literal["evonn", "evonn2", "prism", "topograph", "stratograph", "hybrid", "contenders"]
TaskKind = Literal["classification", "regression", "language_modeling"]


class BenchmarkEntry(BaseModel):
    """Manifest-level benchmark coverage entry."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    task_kind: TaskKind
    metric_name: str
    metric_direction: MetricDirection
    status: BenchmarkStatus


class BudgetEnvelope(BaseModel):
    """Resource budget declared by a run export."""

    model_config = ConfigDict(frozen=True)

    evaluation_count: int
    epochs_per_candidate: int
    effective_training_epochs: int | None = None
    wall_clock_seconds: float | None = None
    generations: int | None = None
    population_size: int | None = None
    budget_policy_name: str | None = None


class DeviceInfo(BaseModel):
    """Execution environment metadata."""

    model_config = ConfigDict(frozen=True)

    device_name: str
    precision_mode: str
    framework: str | None = None
    framework_version: str | None = None


class ArtifactPaths(BaseModel):
    """Artifact references required for downstream comparison and analysis."""

    model_config = ConfigDict(frozen=True)

    config_snapshot: str
    report_markdown: str
    model_summary_json: str | None = None
    genome_summary_json: str | None = None
    dataset_manifest_json: str | None = None
    raw_database: str | None = None


class SearchTelemetry(BaseModel):
    """Optional system-specific search telemetry for correlation and tuning."""

    model_config = ConfigDict(frozen=True)

    qd_enabled: bool = False
    effective_training_epochs: int | None = None
    novelty_weight: float | None = None
    novelty_k: int | None = None
    novelty_archive_limit: int | None = None
    novelty_archive_final_size: int | None = None
    novelty_score_mean: float | None = None
    novelty_score_max: float | None = None
    map_elites_enabled: bool = False
    map_elites_selection_ratio: float | None = None
    map_elites_occupied_niches: int | None = None
    map_elites_total_niches: int | None = None
    map_elites_fill_ratio: float | None = None
    map_elites_insertions: int | None = None
    map_elites_parent_samples: int | None = None


class RunManifest(BaseModel):
    """Top-level run export metadata."""

    model_config = ConfigDict(frozen=True)

    schema_version: str
    system: SystemName
    run_id: str
    run_name: str
    created_at: datetime
    pack_name: str
    seed: int
    benchmarks: list[BenchmarkEntry]
    budget: BudgetEnvelope
    device: DeviceInfo
    artifacts: ArtifactPaths
    search_telemetry: SearchTelemetry | None = None

    @model_validator(mode="after")
    def validate_unique_benchmarks(self) -> "RunManifest":
        benchmark_ids = [entry.benchmark_id for entry in self.benchmarks]
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError("manifest benchmarks must be unique by benchmark_id")
        return self


class ResultRecord(BaseModel):
    """Per-benchmark result record aligned across systems."""

    model_config = ConfigDict(frozen=True)

    system: SystemName
    run_id: str
    benchmark_id: str
    metric_name: str
    metric_direction: MetricDirection
    metric_value: float | None
    quality: float | None = None
    parameter_count: int | None = None
    train_seconds: float | None = None
    peak_memory_mb: float | None = None
    architecture_summary: str | None = None
    genome_id: str | None = None
    status: BenchmarkStatus
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ResultRecord":
        if self.status == "ok" and self.metric_value is None:
            raise ValueError("metric_value is required when status='ok'")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failure_reason is required when status='failed'")
        return self
