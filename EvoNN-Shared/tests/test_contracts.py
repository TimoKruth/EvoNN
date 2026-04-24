from datetime import datetime, timezone

import pytest

from evonn_shared.contracts import (
    ArtifactPaths,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    ResultRecord,
    RunManifest,
)


def test_run_manifest_rejects_duplicate_benchmark_ids() -> None:
    with pytest.raises(ValueError, match="unique by benchmark_id"):
        RunManifest(
            schema_version="1.0",
            system="prism",
            run_id="run-1",
            run_name="run-1",
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            pack_name="tier1_core",
            seed=42,
            benchmarks=[
                BenchmarkEntry(
                    benchmark_id="iris",
                    task_kind="classification",
                    metric_name="accuracy",
                    metric_direction="max",
                    status="ok",
                ),
                BenchmarkEntry(
                    benchmark_id="iris",
                    task_kind="classification",
                    metric_name="accuracy",
                    metric_direction="max",
                    status="ok",
                ),
            ],
            budget=BudgetEnvelope(evaluation_count=64, epochs_per_candidate=20),
            device=DeviceInfo(device_name="apple_silicon", precision_mode="fp32"),
            artifacts=ArtifactPaths(config_snapshot="config.json", report_markdown="report.md"),
        )


def test_result_record_requires_failure_reason_for_failed_status() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        ResultRecord(
            system="prism",
            run_id="run-1",
            benchmark_id="iris",
            metric_name="accuracy",
            metric_direction="max",
            metric_value=None,
            status="failed",
        )
