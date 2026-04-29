"""Core compare/export contracts shared across EvoNN packages."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BenchmarkStatus = Literal["ok", "failed", "skipped", "unsupported", "missing"]
MetricDirection = Literal["max", "min"]
BaselineCoveragePolicy = Literal[
    "required_only_optional_skips_allowed",
    "all_configured_contenders_required",
]
BaselineCoverageStage = Literal["temporary", "steady_state"]
SystemName = Literal[
    "evonn",
    "evonn2",
    "prism",
    "topograph",
    "stratograph",
    "primordia",
    "hybrid",
    "contenders",
]
TaskKind = Literal["classification", "regression", "language_modeling"]
SeedingLadder = Literal["none", "direct", "staged"]
SeedOverlapPolicy = Literal[
    "benchmark-disjoint",
    "benchmark-overlapping",
    "family-overlapping",
    "unknown",
]


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
    actual_evaluations: int | None = None
    cached_evaluations: int | None = None
    failed_evaluations: int | None = None
    invalid_evaluations: int | None = None
    resumed_from_run_id: str | None = None
    resumed_evaluations: int | None = None
    partial_run: bool = False
    evaluation_semantics: str | None = None

    @model_validator(mode="after")
    def validate_accounting_fields(self) -> "BudgetEnvelope":
        for field_name in (
            "evaluation_count",
            "epochs_per_candidate",
            "actual_evaluations",
            "cached_evaluations",
            "failed_evaluations",
            "invalid_evaluations",
            "resumed_evaluations",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if self.resumed_evaluations is not None and self.resumed_from_run_id is None:
            raise ValueError("resumed_from_run_id is required when resumed_evaluations is set")
        return self


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
    contender_summary_json: str | None = None
    dataset_manifest_json: str | None = None
    dataset_manifest_hash: str | None = None
    raw_database: str | None = None
    pack_name: str | None = None
    benchmarks: list[str] | None = None
    canonical_benchmarks: list[str] | None = None


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


class FairnessEnvelope(BaseModel):
    """Explicit fairness metadata carried across run exports."""

    model_config = ConfigDict(frozen=True)

    benchmark_pack_id: str
    seed: int
    evaluation_count: int
    budget_policy_name: str | None = None
    data_signature: str | None = None
    code_version: str | None = None


class SeedingEnvelope(BaseModel):
    """Transfer/seeding provenance carried across compare-grade exports."""

    model_config = ConfigDict(frozen=True)

    seeding_enabled: bool
    seeding_ladder: SeedingLadder
    seed_source_system: SystemName | None = None
    seed_source_run_id: str | None = None
    seed_artifact_path: str | None = None
    seed_target_family: str | None = None
    seed_selected_family: str | None = None
    seed_rank: int | None = None
    seed_overlap_policy: SeedOverlapPolicy | None = None
    representative_genome_id: str | None = None
    representative_architecture_summary: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "SeedingEnvelope":
        if self.seed_rank is not None and self.seed_rank < 1:
            raise ValueError("seed_rank must be >= 1 when provided")
        if self.seeding_enabled:
            if self.seeding_ladder == "none":
                raise ValueError("seeding_ladder must not be 'none' when seeding_enabled is true")
            if self.seed_source_system is None:
                raise ValueError("seed_source_system is required when seeding_enabled is true")
            if self.seed_artifact_path is None:
                raise ValueError("seed_artifact_path is required when seeding_enabled is true")
        elif self.seeding_ladder != "none":
            raise ValueError("seeding_ladder must be 'none' when seeding_enabled is false")
        return self


class BaselineCoverageEnvelope(BaseModel):
    """Optional baseline-completeness policy metadata for fixed-baseline systems."""

    model_config = ConfigDict(frozen=True)

    benchmark_complete_policy: BaselineCoveragePolicy
    policy_stage: BaselineCoverageStage = "temporary"
    policy_reason: str | None = None
    optional_dependency_skips: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


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
    seeding: SeedingEnvelope | None = None
    fairness: FairnessEnvelope | None = None
    baseline_coverage: BaselineCoverageEnvelope | None = None

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
