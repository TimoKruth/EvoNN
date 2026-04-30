import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli.main import app
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
from evonn_compare.orchestration.performance_report import build_performance_report

runner = CliRunner()


def test_build_performance_report_renders_before_after_deltas(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    accepted_root = tmp_path / "accepted"
    scrapped_root = tmp_path / "scrapped"

    _write_dataset(
        baseline_root,
        [
            _measured_row(budget=64, wall_clock=10.0, evals_per_second=8.0),
            _measured_row(budget=256, wall_clock=40.0, evals_per_second=2.0),
        ],
    )
    _write_dataset(
        accepted_root,
        [
            _measured_row(
                budget=64,
                wall_clock=8.0,
                evals_per_second=10.0,
                quality_delta=0.02,
                rank_delta=-0.20,
            ),
            _measured_row(
                budget=256,
                wall_clock=34.0,
                evals_per_second=2.5,
                quality_delta=0.01,
                rank_delta=-0.10,
            ),
        ],
    )
    _write_dataset(
        scrapped_root,
        [
            _measured_row(
                budget=64,
                wall_clock=7.5,
                evals_per_second=10.4,
                quality_status="fail",
                quality_delta=-0.08,
                rank_delta=0.30,
                trust_status="fail",
                observed_state="regressed",
            ),
            _measured_row(
                budget=256,
                wall_clock=32.0,
                evals_per_second=2.8,
                quality_status="fail",
                quality_delta=-0.06,
                rank_delta=0.25,
                trust_status="fail",
                observed_state="regressed",
            ),
        ],
    )

    output_root = tmp_path / "review"
    result = build_performance_report(
        baseline_label="baseline",
        baseline_path=baseline_root,
        candidate_specs=[("vectorized-cache", accepted_root), ("discarded-branch", scrapped_root)],
        outcomes={"vectorized-cache": "accepted", "discarded-branch": "scrapped"},
        compare_label="vectorized-cache",
        output_root=output_root,
    )

    assert result["candidate_count"] == 2
    payload = json.loads((output_root / "performance_report.json").read_text(encoding="utf-8"))
    assert payload["primary_comparison"]["available"] is True
    assert payload["primary_comparison"]["candidate_label"] == "vectorized-cache"
    assert payload["primary_comparison"]["summary"]["verdict"] == "faster-no-regression"
    assert payload["primary_comparison"]["summary"]["candidate_accounting_tags"] == ["full_budget"]
    assert payload["primary_comparison"]["summary"]["median_wall_clock_delta_pct"] == -17.5
    assert payload["primary_comparison"]["summary"]["median_evals_per_second_delta_pct"] == 25.0

    budgets = {row["budget"] for row in payload["primary_comparison"]["deltas"]}
    assert budgets == {64, 256}
    delta_by_budget = {row["budget"]: row for row in payload["primary_comparison"]["deltas"]}
    assert delta_by_budget[64]["wall_clock_delta_pct"] == -20.0
    assert delta_by_budget[64]["evals_per_second_delta_pct"] == 25.0
    assert delta_by_budget[64]["median_quality_delta_vs_baseline"] == 0.02
    assert delta_by_budget[256]["wall_clock_delta_pct"] == -15.0
    assert delta_by_budget[256]["verdict"] == "faster-no-regression"

    history = {row["label"]: row for row in payload["optimization_history"]}
    assert history["vectorized-cache"]["outcome"] == "accepted"
    assert history["discarded-branch"]["outcome"] == "scrapped"
    assert history["discarded-branch"]["quality_regression_count"] == 2
    assert history["discarded-branch"]["trust_regression_count"] == 2
    assert history["discarded-branch"]["verdict"] == "faster-with-guardrail-regressions"

    dashboard_html = (output_root / "performance_dashboard.html").read_text(encoding="utf-8")
    assert "Before/After Delta View" in dashboard_html
    assert "Optimization History" in dashboard_html
    assert "scrapped" in dashboard_html

    report_md = (output_root / "performance_report.md").read_text(encoding="utf-8")
    assert "## Before/After Delta View" in report_md
    assert "| discarded-branch | scrapped | full_budget | faster-with-guardrail-regressions |" in report_md


def test_performance_report_cli_accepts_manifest_and_renders_outputs(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    accepted_root = tmp_path / "accepted"
    scrapped_root = tmp_path / "scrapped"
    output_root = tmp_path / "cli-review"

    _write_dataset(
        baseline_root,
        [_measured_row(budget=64, wall_clock=10.0, evals_per_second=8.0)],
    )
    _write_dataset(
        accepted_root,
        [_measured_row(budget=64, wall_clock=8.0, evals_per_second=10.0, quality_delta=0.02)],
    )
    _write_dataset(
        scrapped_root,
        [
            _measured_row(
                budget=64,
                wall_clock=7.5,
                evals_per_second=10.4,
                quality_status="fail",
                quality_delta=-0.08,
                rank_delta=0.30,
                trust_status="fail",
                observed_state="regressed",
            )
        ],
    )

    result = runner.invoke(
        app,
        [
            "performance-report",
            str(baseline_root / "baseline_manifest.json"),
            "--candidate",
            f"accepted-opt={accepted_root}",
            "--candidate",
            f"scrapped-opt={scrapped_root}",
            "--outcome",
            "accepted-opt=accepted",
            "--outcome",
            "scrapped-opt=scrapped",
            "--compare-label",
            "accepted-opt",
            "--output-root",
            str(output_root),
        ],
    )

    assert result.exit_code == 0
    assert f"report_json\t{output_root / 'performance_report.json'}" in result.stdout
    assert f"dashboard_html\t{output_root / 'performance_dashboard.html'}" in result.stdout
    payload = json.loads((output_root / "performance_report.json").read_text(encoding="utf-8"))
    assert payload["primary_comparison"]["candidate_label"] == "accepted-opt"
    assert {row["outcome"] for row in payload["optimization_history"]} == {"accepted", "scrapped"}


def _write_dataset(root: Path, rows: list[PerformanceRow]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    perf_rows_path = root / "perf_rows.jsonl"
    perf_rows_path.write_text(
        "".join(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = PerformanceBaselineManifest(
        generated_at="2026-04-30T12:00:00+00:00",
        git_sha="abc1234",
        baseline_root=str(root),
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
        seeds=[42],
        cache_modes=["cold"],
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
        planned_case_count=len(rows),
        status_counts={"measured": len(rows)},
        system_counts=[
            PerformanceSystemCoverage(
                system="prism",
                planned_case_count=len(rows),
                backends=[
                    PerformanceSystemBackendSummary(
                        backend_class="mlx_truth",
                        backend_label="mlx",
                        host_label="macos",
                        planned_case_count=len(rows),
                    )
                ],
            )
        ],
        artifacts=PerformanceBaselineArtifacts(
            raw_runs=str(root / "raw_runs"),
            perf_rows=str(perf_rows_path),
            baseline_summary=str(root / "baseline_summary.md"),
            perf_dashboard=str(root / "perf_dashboard.html"),
            perf_dashboard_json=str(root / "perf_dashboard.json"),
        ),
        review_references=PerformanceReviewReferences(
            workflow_doc="PERFORMANCE_OPTIMIZATION_WORKFLOW.md",
            pull_request_template=".github/pull_request_template.md",
            child_issue_template="PERFORMANCE_OPTIMIZATION_WORKFLOW.md#optimization-child-issue-template",
            branch_outcome_recording="PERFORMANCE_OPTIMIZATION_WORKFLOW.md#branch-outcome-recording",
        ),
    )
    (root / "baseline_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _measured_row(
    *,
    budget: int,
    wall_clock: float,
    evals_per_second: float,
    quality_status: str = "pass",
    quality_delta: float = 0.0,
    rank_delta: float = 0.0,
    trust_status: str = "pass",
    observed_state: str = "same",
) -> PerformanceRow:
    return PerformanceRow(
        record_type="measured_performance_baseline_case",
        status="measured",
        generated_at="2026-04-30T12:00:00+00:00",
        git_sha="abc1234",
        case_id=f"prism__mlx__tier1_core__eval{budget}__seed42__cold",
        system="prism",
        backend_class="mlx_truth",
        backend_label="mlx",
        host_label="macos",
        pack_name="tier1_core",
        pack_path="/tmp/tier1_core.yaml",
        pack_tier=1,
        benchmark_count=12,
        budget=budget,
        seed=42,
        cache_mode="cold",
        accounting_tags=("full_budget",),
        raw_run_dir=f"/tmp/raw/prism/eval{budget}",
        metrics=PerformanceMetrics(
            wall_clock_seconds=wall_clock,
            evals_per_second=evals_per_second,
            cache_hit_rate=0.5,
            reuse_rate=0.4,
            failure_count=0,
        ),
        quality_guard=PerformanceQualityGuard(
            status=quality_status,
            median_rank=1.0,
            median_rank_delta_vs_baseline=rank_delta,
            quality_delta_vs_baseline=quality_delta,
        ),
        trust_guard=PerformanceTrustGuard(
            required_state="same-or-better",
            observed_state=observed_state,
            status=trust_status,
        ),
        notes=["measured run"],
    )
