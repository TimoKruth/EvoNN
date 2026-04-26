"""Stable compatibility re-exports for compare/export contracts.

The canonical contract models live in :mod:`evonn_shared.contracts`.
This module is intentionally retained as the public EvoNN-Compare import surface
for downstream callers that still import ``evonn_compare.contracts.models``.

Internal EvoNN-Compare runtime code should prefer direct imports from
``evonn_shared.contracts`` so shared-substrate ownership stays explicit.
"""

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

__all__ = [
    "ArtifactPaths",
    "BenchmarkEntry",
    "BudgetEnvelope",
    "DeviceInfo",
    "FairnessEnvelope",
    "ResultRecord",
    "RunManifest",
    "SearchTelemetry",
]
