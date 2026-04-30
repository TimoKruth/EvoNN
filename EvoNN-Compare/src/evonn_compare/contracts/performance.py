"""Performance artifact contracts owned by EvoNN-Compare."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from evonn_shared.contracts import BudgetAccountingTag


class PerformanceBackendTarget(BaseModel):
    """One supported backend lane for a system in the performance matrix."""

    model_config = ConfigDict(frozen=True)

    backend_class: str
    backend_label: str
    host_label: str


class PerformanceMetrics(BaseModel):
    """Normalized runtime metrics for one performance row."""

    model_config = ConfigDict(frozen=True)

    wall_clock_seconds: float | None = None
    evals_per_second: float | None = None
    train_seconds: float | None = None
    data_load_seconds: float | None = None
    cache_hit_rate: float | None = None
    reuse_rate: float | None = None
    failure_count: int | None = None
    peak_memory_mb: float | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "PerformanceMetrics":
        for field_name in (
            "wall_clock_seconds",
            "evals_per_second",
            "train_seconds",
            "data_load_seconds",
            "failure_count",
            "peak_memory_mb",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        for field_name in ("cache_hit_rate", "reuse_rate"):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        return self


class PerformanceQualityGuard(BaseModel):
    """Quality guard fields that gate performance-branch acceptance."""

    model_config = ConfigDict(frozen=True)

    status: str
    median_rank: float | None = None
    median_rank_delta_vs_baseline: float | None = None
    quality_delta_vs_baseline: float | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "PerformanceQualityGuard":
        if self.median_rank is not None and self.median_rank < 0:
            raise ValueError("median_rank must be >= 0")
        return self


class PerformanceTrustGuard(BaseModel):
    """Trust-state guard fields for performance rows."""

    model_config = ConfigDict(frozen=True)

    required_state: str
    observed_state: str | None = None
    status: str

    @model_validator(mode="after")
    def validate_shape(self) -> "PerformanceTrustGuard":
        if self.status == "pending" and self.observed_state is not None:
            raise ValueError("observed_state must be empty while trust_guard status is pending")
        if self.status != "pending" and self.observed_state is None:
            raise ValueError("observed_state is required once trust_guard status is not pending")
        return self


class PerformanceRow(BaseModel):
    """Normalized JSONL row for planned or measured performance artifacts."""

    model_config = ConfigDict(frozen=True)

    record_type: Literal[
        "planned_performance_baseline_case",
        "measured_performance_baseline_case",
    ]
    status: Literal["planned", "measured", "failed"]
    generated_at: str
    git_sha: str
    case_id: str
    system: str
    backend_class: str
    backend_label: str
    host_label: str
    pack_name: str
    pack_path: str
    pack_tier: Literal[1, 2, 3]
    benchmark_count: int
    budget: int
    seed: int
    cache_mode: str
    accounting_tags: tuple[BudgetAccountingTag, ...] = ()
    actual_evaluations: int | None = None
    cached_evaluations: int | None = None
    resumed_evaluations: int | None = None
    screened_evaluations: int | None = None
    deduplicated_evaluations: int | None = None
    reduced_fidelity_evaluations: int | None = None
    worker_count: int | None = None
    precision: str | None = None
    device: str | None = None
    raw_run_dir: str
    metrics: PerformanceMetrics
    quality_guard: PerformanceQualityGuard
    trust_guard: PerformanceTrustGuard
    notes: list[str]

    @model_validator(mode="after")
    def validate_semantics(self) -> "PerformanceRow":
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if self.benchmark_count < 1:
            raise ValueError("benchmark_count must be >= 1")
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if not self.raw_run_dir:
            raise ValueError("raw_run_dir must not be empty")
        if not self.notes:
            raise ValueError("notes must include at least one entry")
        for field_name in (
            "actual_evaluations",
            "cached_evaluations",
            "resumed_evaluations",
            "screened_evaluations",
            "deduplicated_evaluations",
            "reduced_fidelity_evaluations",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if len(set(self.accounting_tags)) != len(self.accounting_tags):
            raise ValueError("accounting_tags must not contain duplicates")

        metrics_present = any(value is not None for value in self.metrics.model_dump().values())
        if self.status == "planned":
            if self.record_type != "planned_performance_baseline_case":
                raise ValueError("planned rows must use the planned record_type")
            if metrics_present:
                raise ValueError("planned rows must not include measured metrics")
            if self.quality_guard.status != "pending":
                raise ValueError("planned rows must keep quality_guard status pending")
            if self.trust_guard.status != "pending":
                raise ValueError("planned rows must keep trust_guard status pending")
        else:
            if self.record_type != "measured_performance_baseline_case":
                raise ValueError("measured and failed rows must use the measured record_type")
            if not metrics_present:
                raise ValueError("measured and failed rows must include at least one metric")

        if self.status == "failed" and (self.metrics.failure_count or 0) < 1:
            raise ValueError("failed rows must report a positive failure_count")
        return self


class PerformancePackCoverage(BaseModel):
    """One pack covered by a performance baseline manifest."""

    model_config = ConfigDict(frozen=True)

    pack_name: str
    pack_path: str
    tier: int
    benchmark_count: int
    default_budget: int


class PerformanceSystemBackendSummary(BaseModel):
    """Per-system backend coverage in the baseline manifest."""

    model_config = ConfigDict(frozen=True)

    backend_class: str
    backend_label: str
    host_label: str
    planned_case_count: int


class PerformanceSystemCoverage(BaseModel):
    """Per-system planned coverage summary in the baseline manifest."""

    model_config = ConfigDict(frozen=True)

    system: str
    planned_case_count: int
    backends: list[PerformanceSystemBackendSummary]


class PerformanceBaselineArtifacts(BaseModel):
    """Artifact paths emitted by the baseline planner."""

    model_config = ConfigDict(frozen=True)

    raw_runs: str
    perf_rows: str
    baseline_summary: str
    perf_dashboard: str
    perf_dashboard_json: str


class PerformanceReviewReferences(BaseModel):
    """Canonical repo-owned review references for optimization branches."""

    model_config = ConfigDict(frozen=True)

    workflow_doc: str
    pull_request_template: str
    child_issue_template: str
    branch_outcome_recording: str


class PerformanceBaselineManifest(BaseModel):
    """Canonical manifest for a performance baseline workspace."""

    model_config = ConfigDict(frozen=True)

    generated_at: str
    git_sha: str
    baseline_root: str
    packs: list[PerformancePackCoverage]
    budgets: list[int]
    seeds: list[int]
    cache_modes: list[str]
    systems: list[str]
    supported_backends: dict[str, list[PerformanceBackendTarget]]
    planned_case_count: int
    status_counts: dict[str, int]
    system_counts: list[PerformanceSystemCoverage]
    artifacts: PerformanceBaselineArtifacts
    review_references: PerformanceReviewReferences

    @model_validator(mode="after")
    def validate_summary_counts(self) -> "PerformanceBaselineManifest":
        if not self.packs:
            raise ValueError("packs must not be empty")
        if not self.budgets:
            raise ValueError("budgets must not be empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if not self.cache_modes:
            raise ValueError("cache_modes must not be empty")
        if not self.systems:
            raise ValueError("systems must not be empty")
        if not self.system_counts:
            raise ValueError("system_counts must not be empty")
        if self.planned_case_count < 1:
            raise ValueError("planned_case_count must be >= 1")

        status_total = sum(self.status_counts.values())
        if status_total != self.planned_case_count:
            raise ValueError("status_counts must sum to planned_case_count")

        supported_systems = set(self.supported_backends)
        listed_systems = set(self.systems)
        counted_systems = {entry.system for entry in self.system_counts}
        if listed_systems != supported_systems or listed_systems != counted_systems:
            raise ValueError("systems, supported_backends, and system_counts must describe the same systems")

        if len(self.systems) != len(listed_systems):
            raise ValueError("systems must not contain duplicates")

        planned_total = 0
        for entry in self.system_counts:
            supported_backends = self.supported_backends.get(entry.system)
            if supported_backends is None:
                raise ValueError(f"unsupported system count entry: {entry.system}")
            supported_backend_classes = {backend.backend_class for backend in supported_backends}
            counted_backend_classes = {backend.backend_class for backend in entry.backends}
            if supported_backend_classes != counted_backend_classes:
                raise ValueError(
                    f"system_counts backends for {entry.system} must match supported_backends"
                )
            backend_total = sum(backend.planned_case_count for backend in entry.backends)
            if backend_total != entry.planned_case_count:
                raise ValueError(
                    f"system_counts planned_case_count for {entry.system} must equal backend subtotal"
                )
            planned_total += entry.planned_case_count

        if planned_total != self.planned_case_count:
            raise ValueError("system_counts must sum to planned_case_count")
        return self


__all__ = [
    "PerformanceBackendTarget",
    "PerformanceBaselineArtifacts",
    "PerformanceBaselineManifest",
    "PerformanceMetrics",
    "PerformancePackCoverage",
    "PerformanceQualityGuard",
    "PerformanceReviewReferences",
    "PerformanceRow",
    "PerformanceSystemBackendSummary",
    "PerformanceSystemCoverage",
    "PerformanceTrustGuard",
]
