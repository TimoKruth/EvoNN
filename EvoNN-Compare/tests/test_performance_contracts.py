import json

import pytest

from evonn_compare.contracts.performance import (
    PerformanceBackendTarget,
    PerformanceBaselineArtifacts,
    PerformanceBaselineManifest,
    PerformanceMetrics,
    PerformancePackCoverage,
    PerformanceQualityGuard,
    PerformanceReviewReferences,
    PerformanceRow,
    PerformanceSystemBackendSummary,
    PerformanceSystemCoverage,
    PerformanceTrustGuard,
)


def test_performance_row_contract_round_trips() -> None:
    row = PerformanceRow(
        record_type="planned_performance_baseline_case",
        status="planned",
        generated_at="2026-04-30T09:00:00+00:00",
        git_sha="abc1234",
        case_id="prism__mlx__tier1_core__eval64__seed42__cold",
        system="prism",
        backend_class="mlx_truth",
        backend_label="mlx",
        host_label="macos",
        pack_name="tier1_core",
        pack_path="/tmp/tier1_core.yaml",
        pack_tier=1,
        benchmark_count=12,
        budget=64,
        seed=42,
        cache_mode="cold",
        accounting_tags=("full_budget",),
        raw_run_dir="/tmp/raw/prism",
        metrics=PerformanceMetrics(),
        quality_guard=PerformanceQualityGuard(status="pending"),
        trust_guard=PerformanceTrustGuard(required_state="same-or-better", status="pending"),
        notes=["planned only"],
    )

    payload = row.model_dump(mode="json")
    round_tripped = PerformanceRow.model_validate_json(json.dumps(payload))

    assert round_tripped.case_id == row.case_id
    assert round_tripped.accounting_tags == ("full_budget",)
    assert round_tripped.metrics.wall_clock_seconds is None
    assert round_tripped.trust_guard.required_state == "same-or-better"


def test_performance_manifest_contract_round_trips() -> None:
    manifest = PerformanceBaselineManifest(
        generated_at="2026-04-30T09:00:00+00:00",
        git_sha="abc1234",
        baseline_root="/tmp/performance-baseline",
        packs=[
            PerformancePackCoverage(
                pack_name="tier1_core",
                pack_path="/tmp/tier1_core.yaml",
                tier=1,
                benchmark_count=12,
                default_budget=64,
            )
        ],
        budgets=[64, 256],
        seeds=[42, 43],
        cache_modes=["cold", "warm"],
        systems=["prism"],
        supported_backends={
            "prism": [
                PerformanceBackendTarget(
                    backend_class="mlx_truth",
                    backend_label="mlx",
                    host_label="macos",
                )
            ]
        },
        planned_case_count=4,
        status_counts={"planned": 4},
        system_counts=[
            PerformanceSystemCoverage(
                system="prism",
                planned_case_count=4,
                backends=[
                    PerformanceSystemBackendSummary(
                        backend_class="mlx_truth",
                        backend_label="mlx",
                        host_label="macos",
                        planned_case_count=4,
                    )
                ],
            )
        ],
        artifacts=PerformanceBaselineArtifacts(
            raw_runs="/tmp/performance-baseline/raw_runs",
            perf_rows="/tmp/performance-baseline/perf_rows.jsonl",
            baseline_summary="/tmp/performance-baseline/baseline_summary.md",
            perf_dashboard="/tmp/performance-baseline/perf_dashboard.html",
            perf_dashboard_json="/tmp/performance-baseline/perf_dashboard.json",
        ),
        review_references=PerformanceReviewReferences(
            workflow_doc="PERFORMANCE_OPTIMIZATION_WORKFLOW.md",
            pull_request_template=".github/pull_request_template.md",
            child_issue_template="PERFORMANCE_OPTIMIZATION_WORKFLOW.md#optimization-child-issue-template",
            branch_outcome_recording="PERFORMANCE_OPTIMIZATION_WORKFLOW.md#branch-outcome-recording",
        ),
    )

    payload = manifest.model_dump(mode="json")
    round_tripped = PerformanceBaselineManifest.model_validate_json(json.dumps(payload))

    assert round_tripped.artifacts.perf_rows.endswith("perf_rows.jsonl")
    assert round_tripped.system_counts[0].backends[0].backend_label == "mlx"
    assert round_tripped.review_references.workflow_doc == "PERFORMANCE_OPTIMIZATION_WORKFLOW.md"


def test_planned_row_rejects_measured_metrics() -> None:
    with pytest.raises(ValueError, match="planned rows must not include measured metrics"):
        PerformanceRow(
            record_type="planned_performance_baseline_case",
            status="planned",
            generated_at="2026-04-30T09:00:00+00:00",
            git_sha="abc1234",
            case_id="stratograph__fallback__tier1_core__eval64__seed42__cold",
            system="stratograph",
            backend_class="linux_fallback",
            backend_label="fallback",
            host_label="linux",
            pack_name="tier1_core",
            pack_path="/tmp/tier1_core.yaml",
            pack_tier=1,
            benchmark_count=12,
            budget=64,
            seed=42,
            cache_mode="cold",
            raw_run_dir="/tmp/raw/stratograph",
            metrics=PerformanceMetrics(wall_clock_seconds=12.5),
            quality_guard=PerformanceQualityGuard(status="pending"),
            trust_guard=PerformanceTrustGuard(required_state="same-or-better", status="pending"),
            notes=["planned only"],
        )


def test_failed_row_requires_failure_count() -> None:
    with pytest.raises(ValueError, match="failed rows must report a positive failure_count"):
        PerformanceRow(
            record_type="measured_performance_baseline_case",
            status="failed",
            generated_at="2026-04-30T09:00:00+00:00",
            git_sha="abc1234",
            case_id="primordia__fallback__tier1_core__eval64__seed42__cold",
            system="primordia",
            backend_class="linux_fallback",
            backend_label="fallback",
            host_label="linux",
            pack_name="tier1_core",
            pack_path="/tmp/tier1_core.yaml",
            pack_tier=1,
            benchmark_count=12,
            budget=64,
            seed=42,
            cache_mode="cold",
            raw_run_dir="/tmp/raw/primordia",
            metrics=PerformanceMetrics(wall_clock_seconds=18.0),
            quality_guard=PerformanceQualityGuard(status="warn"),
            trust_guard=PerformanceTrustGuard(
                required_state="same-or-better",
                observed_state="regressed",
                status="fail",
            ),
            notes=["run failed"],
        )


def test_trust_guard_pending_rejects_observed_state() -> None:
    with pytest.raises(ValueError, match="observed_state must be empty"):
        PerformanceTrustGuard(
            required_state="same-or-better",
            observed_state="same",
            status="pending",
        )


def test_metrics_reject_invalid_rate() -> None:
    with pytest.raises(ValueError, match="cache_hit_rate must be between 0.0 and 1.0"):
        PerformanceMetrics(cache_hit_rate=1.5)


def test_manifest_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="status_counts must sum to planned_case_count"):
        PerformanceBaselineManifest(
            generated_at="2026-04-30T09:00:00+00:00",
            git_sha="abc1234",
            baseline_root="/tmp/performance-baseline",
            packs=[
                PerformancePackCoverage(
                    pack_name="tier1_core",
                    pack_path="/tmp/tier1_core.yaml",
                    tier=1,
                    benchmark_count=12,
                    default_budget=64,
                )
            ],
            budgets=[64],
            seeds=[42],
            cache_modes=["cold"],
            systems=["contenders"],
            supported_backends={
                "contenders": [
                    PerformanceBackendTarget(
                        backend_class="linux_fallback",
                        backend_label="sklearn",
                        host_label="linux",
                    )
                ]
            },
            planned_case_count=2,
            status_counts={"planned": 1},
            system_counts=[
                PerformanceSystemCoverage(
                    system="contenders",
                    planned_case_count=2,
                    backends=[
                        PerformanceSystemBackendSummary(
                            backend_class="linux_fallback",
                            backend_label="sklearn",
                            host_label="linux",
                            planned_case_count=2,
                        )
                    ],
                )
            ],
            artifacts=PerformanceBaselineArtifacts(
                raw_runs="/tmp/performance-baseline/raw_runs",
                perf_rows="/tmp/performance-baseline/perf_rows.jsonl",
                baseline_summary="/tmp/performance-baseline/baseline_summary.md",
                perf_dashboard="/tmp/performance-baseline/perf_dashboard.html",
                perf_dashboard_json="/tmp/performance-baseline/perf_dashboard.json",
            ),
            review_references=PerformanceReviewReferences(
                workflow_doc="PERFORMANCE_OPTIMIZATION_WORKFLOW.md",
                pull_request_template=".github/pull_request_template.md",
                child_issue_template="PERFORMANCE_OPTIMIZATION_WORKFLOW.md#optimization-child-issue-template",
                branch_outcome_recording="PERFORMANCE_OPTIMIZATION_WORKFLOW.md#branch-outcome-recording",
            ),
        )
