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
from evonn_shared.lm_cache import (
    LMCacheSpec,
    default_lm_cache_spec,
    generate_lm_cache,
    validate_default_lm_cache,
    validate_lm_cache,
)
from evonn_shared.runs import RunCoordinates
from evonn_shared.training import (
    calibrate_regression_predictions,
    regression_target_stats,
    restore_regression_predictions,
    standardize_regression_targets,
)

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
    "LMCacheSpec",
    "PerformanceEnvelope",
    "ResultRecord",
    "RunCoordinates",
    "RuntimeEnvelope",
    "RunManifest",
    "SearchTelemetry",
    "TaskKind",
    "benchmark_signature",
    "calibrate_regression_predictions",
    "default_artifact",
    "default_data_signature",
    "fairness_manifest",
    "default_lm_cache_spec",
    "generate_lm_cache",
    "regression_target_stats",
    "restore_regression_predictions",
    "standardize_regression_targets",
    "summary_core_from_results",
    "validate_default_lm_cache",
    "validate_lm_cache",
    "write_json",
]
