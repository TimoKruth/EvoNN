from datetime import datetime, timezone

import pytest

from evonn_shared.contracts import (
    ArtifactPaths,
    BaselineCoverageEnvelope,
    BenchmarkEntry,
    BudgetEnvelope,
    DeviceInfo,
    ResultRecord,
    RunManifest,
    SeedingEnvelope,
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


def test_budget_envelope_requires_resume_source_for_resumed_evaluations() -> None:
    with pytest.raises(ValueError, match="resumed_from_run_id"):
        BudgetEnvelope(
            evaluation_count=64,
            epochs_per_candidate=20,
            resumed_evaluations=5,
        )


def test_run_manifest_accepts_baseline_coverage_policy() -> None:
    manifest = RunManifest(
        schema_version="1.0",
        system="contenders",
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
            )
        ],
        budget=BudgetEnvelope(evaluation_count=64, epochs_per_candidate=20),
        device=DeviceInfo(device_name="linux_x86_64", precision_mode="fp32"),
        artifacts=ArtifactPaths(config_snapshot="config.json", report_markdown="report.md"),
        baseline_coverage=BaselineCoverageEnvelope(
            benchmark_complete_policy="required_only_optional_skips_allowed",
            policy_stage="steady_state",
            policy_reason="shared contender floor is sklearn-backed; boosted extras widen coverage when available",
            optional_dependency_skips={"tabular": ("xgb_small", "lgbm_small")},
            notes=("optional dependency backends skipped by policy",),
        ),
    )

    assert manifest.baseline_coverage is not None
    assert manifest.baseline_coverage.benchmark_complete_policy == "required_only_optional_skips_allowed"
    assert manifest.baseline_coverage.policy_stage == "steady_state"


def test_run_manifest_accepts_seeding_envelope() -> None:
    manifest = RunManifest(
        schema_version="1.0",
        system="topograph",
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
            )
        ],
        budget=BudgetEnvelope(evaluation_count=64, epochs_per_candidate=20),
        device=DeviceInfo(device_name="apple_silicon", precision_mode="fp32"),
        artifacts=ArtifactPaths(config_snapshot="config.json", report_markdown="report.md"),
        seeding=SeedingEnvelope(
            seeding_enabled=True,
            seeding_ladder="direct",
            seed_source_system="primordia",
            seed_source_run_id="prim-run-7",
            seed_artifact_path="seed_candidates.json",
            seed_target_family="tabular",
            seed_selected_family="sparse_mlp",
            seed_rank=1,
            seed_overlap_policy="family-overlapping",
        ),
    )

    assert manifest.seeding is not None
    assert manifest.seeding.seed_source_system == "primordia"


def test_seeding_envelope_rejects_missing_artifact_path_for_seeded_runs() -> None:
    with pytest.raises(ValueError, match="seed_artifact_path"):
        SeedingEnvelope(
            seeding_enabled=True,
            seeding_ladder="direct",
            seed_source_system="primordia",
        )
