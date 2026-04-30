from __future__ import annotations

from datetime import datetime, timezone

from evonn_compare.contracts.parity import BudgetPolicy, ParityBenchmark, ParityPack, SeedPolicy
from evonn_compare.contracts.validation import validate_contract
from evonn_shared.contracts import ArtifactPaths, BenchmarkEntry, BudgetEnvelope, DeviceInfo, ResultRecord, RunManifest


def _pack() -> ParityPack:
    return ParityPack(
        name="tier1_core_smoke",
        tier=1,
        description="test pack",
        budget_policy=BudgetPolicy(
            evaluation_count=16,
            epochs_per_candidate=1,
            budget_tolerance_pct=0.0,
        ),
        seed_policy=SeedPolicy(mode="shared", required=True),
        benchmarks=[
            ParityBenchmark(
                benchmark_id="iris_classification",
                metric_name="accuracy",
                metric_direction="max",
                native_ids={"prism": "iris_classification"},
                task_kind="classification",
            )
        ],
    )


def _manifest(
    *,
    actual_evaluations: int | None = None,
    cached_evaluations: int | None = None,
    partial_run: bool = False,
    evaluation_semantics: str | None = None,
) -> RunManifest:
    return RunManifest(
        schema_version="1.0",
        system="prism",
        run_id="run-1",
        run_name="run-1",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        pack_name="tier1_core_smoke",
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
            evaluation_count=16,
            epochs_per_candidate=1,
            actual_evaluations=actual_evaluations,
            cached_evaluations=cached_evaluations,
            partial_run=partial_run,
            evaluation_semantics=evaluation_semantics,
        ),
        device=DeviceInfo(device_name="apple_silicon", precision_mode="fp32"),
        artifacts=ArtifactPaths(config_snapshot="config.json", report_markdown="report.md"),
    )


def _results() -> list[ResultRecord]:
    return [
        ResultRecord(
            system="prism",
            run_id="run-1",
            benchmark_id="iris_classification",
            metric_name="accuracy",
            metric_direction="max",
            metric_value=0.9,
            status="ok",
        )
    ]


def test_validate_contract_warns_when_budget_accounting_is_missing() -> None:
    report = validate_contract(_manifest(), _results(), _pack())

    assert {issue.code for issue in report.issues} >= {
        "budget_actual_evaluations_missing",
        "budget_semantics_missing",
    }


def test_validate_contract_warns_when_actual_budget_is_lower_without_partial_flag() -> None:
    report = validate_contract(
        _manifest(actual_evaluations=8, partial_run=False, evaluation_semantics="one evolved candidate evaluation"),
        _results(),
        _pack(),
    )

    assert any(issue.code == "budget_actual_lt_declared" for issue in report.issues)


def test_validate_contract_does_not_warn_when_partial_run_is_explicit() -> None:
    report = validate_contract(
        _manifest(actual_evaluations=8, partial_run=True, evaluation_semantics="one evolved candidate evaluation"),
        _results(),
        _pack(),
    )

    assert all(issue.code != "budget_actual_lt_declared" for issue in report.issues)


def test_validate_contract_accepts_cached_budget_accounting_when_totals_match() -> None:
    report = validate_contract(
        _manifest(
            actual_evaluations=0,
            cached_evaluations=16,
            partial_run=False,
            evaluation_semantics="one contender fit/eval pass counted per contender in the configured pool",
        ),
        _results(),
        _pack(),
    )

    assert all(issue.code != "budget_actual_lt_declared" for issue in report.issues)
