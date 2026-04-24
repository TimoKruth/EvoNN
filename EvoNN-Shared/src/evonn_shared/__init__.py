"""Shared substrate contracts and helpers for EvoNN."""

from evonn_shared.benchmarks import BenchmarkDescriptor, MetricDirection, TaskKind
from evonn_shared.budgets import BudgetContract
from evonn_shared.contracts import (
    ArtifactPaths,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    FairnessEnvelope,
    ResultRecord,
    RunManifest,
    SearchTelemetry,
)
from evonn_shared.runs import RunCoordinates

__all__ = [
    "ArtifactPaths",
    "BenchmarkDescriptor",
    "BenchmarkEntry",
    "BudgetContract",
    "BudgetEnvelope",
    "DeviceInfo",
    "FairnessEnvelope",
    "MetricDirection",
    "ResultRecord",
    "RunCoordinates",
    "RunManifest",
    "SearchTelemetry",
    "TaskKind",
]
