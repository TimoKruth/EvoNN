from datetime import datetime, timezone

from evonn_compare.contracts.models import (
    ArtifactPaths,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    RunManifest,
    SearchTelemetry,
)
from evonn_compare.reporting.compare_md import _telemetry_row


def test_telemetry_row_preserves_zero_effective_epochs() -> None:
    manifest = RunManifest(
        schema_version="1.0",
        system="topograph",
        run_id="topograph-run",
        run_name="topograph-run",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        pack_name="pack",
        seed=42,
        benchmarks=[
            BenchmarkEntry(
                benchmark_id="iris_classification",
                task_kind="classification",
                metric_name="accuracy",
                metric_direction="max",
                status="ok",
            )
        ],
        budget=BudgetEnvelope(
            evaluation_count=64,
            epochs_per_candidate=20,
            effective_training_epochs=7,
        ),
        device=DeviceInfo(device_name="apple_silicon", precision_mode="fp32"),
        artifacts=ArtifactPaths(
            config_snapshot="config_snapshot.json",
            report_markdown="report.md",
        ),
        search_telemetry=SearchTelemetry(
            qd_enabled=True,
            effective_training_epochs=0,
        ),
    )

    row = _telemetry_row(manifest)

    assert "| 0 | yes |" in row
