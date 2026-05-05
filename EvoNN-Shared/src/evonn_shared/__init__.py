"""Shared substrate contracts and helpers for EvoNN."""

from evonn_shared.benchmarks import BenchmarkDescriptor, MetricDirection, TaskKind
from evonn_shared.budgets import BudgetContract
from evonn_shared.contracts import (
    ArtifactPaths,
    ArtifactCompletenessEnvelope,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    DiagnosticsEnvelope,
    FairnessEnvelope,
    PerformanceEnvelope,
    ResultRecord,
    RuntimeEnvelope,
    RunManifest,
    SearchTelemetry,
)
from evonn_shared.manifests import (
    benchmark_signature,
    default_artifact,
    default_data_signature,
    fairness_manifest,
    summary_core_from_results,
    write_json,
)
from evonn_shared.runs import RunCoordinates

__all__ = [
    "ArtifactPaths",
    "ArtifactCompletenessEnvelope",
    "BenchmarkDescriptor",
    "BenchmarkEntry",
    "BudgetContract",
    "BudgetEnvelope",
    "DeviceInfo",
    "DiagnosticsEnvelope",
    "FairnessEnvelope",
    "MetricDirection",
    "PerformanceEnvelope",
    "ResultRecord",
    "RunCoordinates",
    "RuntimeEnvelope",
    "RunManifest",
    "SearchTelemetry",
    "TaskKind",
    "benchmark_signature",
    "default_artifact",
    "default_data_signature",
    "fairness_manifest",
    "summary_core_from_results",
    "write_json",
]
