from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_primordia.cli import app


runner = CliRunner()


def test_inspect_renders_compact_run_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "sample_run",
                "run_name": "sample_run",
                "runtime": "numpy-fallback",
                "runtime_version": "fallback-0.9",
                "evaluation_count": 6,
                "target_evaluation_count": 8,
                "benchmark_count": 2,
                "failure_count": 1,
                "wall_clock_seconds": 12.5,
                "primitive_usage": {"mlp": 4, "embedding": 2},
                "group_counts": {"tabular": 1, "language_modeling": 1},
                "best_results": [
                    {
                        "benchmark_name": "moons",
                        "primitive_name": "mlp",
                        "metric_name": "accuracy",
                        "metric_value": 0.91,
                        "status": "ok",
                    },
                    {
                        "benchmark_name": "tiny_lm_synthetic",
                        "primitive_name": "embedding",
                        "metric_name": "loss",
                        "metric_value": 1.2,
                        "status": "failed",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "primitive_bank_summary.json").write_text(
        json.dumps(
            {
                "primitive_families": [
                    {
                        "family": "mlp",
                        "evaluation_count": 4,
                        "benchmark_wins": 2,
                        "benchmarks_won": ["moons", "iris"],
                        "representative_architecture_summary": "mlp[64,32]",
                    },
                    {
                        "family": "embedding",
                        "evaluation_count": 2,
                        "benchmark_wins": 0,
                        "benchmarks_won": [],
                        "representative_architecture_summary": "embedding[128]",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "tiny_lm_synthetic",
                    "primitive_name": "embedding",
                    "primitive_family": "embedding",
                    "status": "failed",
                    "failure_reason": "OOM during token embedding warmup",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "sample_run" in result.output
    assert "numpy-fallback" in result.output
    assert "fallback-0.9" in result.output
    assert "Evaluation Count" in result.output
    assert "Target Evaluations" in result.output
    assert "Primitive Usage" in result.output
    assert "Benchmark Group Coverage" in result.output
    assert "language_modeling" in result.output
    assert "Benchmark Wins" in result.output
    assert "Representative" in result.output
    assert "Architecture" in result.output
    assert "Recent Failures" in result.output
    assert "OOM during token embedding warmup" in result.output
    assert "Best Benchmarks" in result.output
    assert "moons" in result.output
    assert "iris" in result.output
    assert "mlp[64,32]" in result.output


def test_inspect_rebuilds_primitive_bank_from_summary_and_trials_when_bank_artifact_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "rebuild_bank"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "rebuild_bank",
                "run_name": "rebuild_bank",
                "runtime": "numpy-fallback",
                "evaluation_count": 3,
                "benchmark_count": 2,
                "failure_count": 0,
                "primitive_usage": {"mlp": 2, "embedding": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_results.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "moons",
                    "primitive_name": "mlp-wide",
                    "primitive_family": "mlp",
                    "metric_name": "accuracy",
                    "metric_value": 0.93,
                    "status": "ok",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trial_records.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "moons",
                    "primitive_name": "mlp-wide",
                    "primitive_family": "mlp",
                    "metric_name": "accuracy",
                    "metric_value": 0.93,
                    "quality": 0.93,
                    "genome_id": "mlp-wide-1",
                    "architecture_summary": "mlp[32,16]",
                    "status": "ok",
                },
                {
                    "benchmark_name": "tiny_lm_synthetic",
                    "primitive_name": "embedding-compact",
                    "primitive_family": "embedding",
                    "metric_name": "loss",
                    "metric_value": 1.4,
                    "quality": -1.4,
                    "genome_id": "embedding-compact-1",
                    "architecture_summary": "embedding[16]",
                    "status": "ok",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "Primitive Bank" in result.output
    assert "Benchmark Wins" in result.output
    assert "mlp" in result.output
    assert "embedding" in result.output
    assert "moons" in result.output


def test_inspect_reads_best_benchmarks_from_best_results_artifact_when_summary_omits_them(tmp_path: Path) -> None:
    run_dir = tmp_path / "best_results_artifact"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "best_results_artifact",
                "runtime": "mlx",
                "evaluation_count": 2,
                "benchmark_count": 1,
                "failure_count": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_results.json").write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "moons",
                    "primitive_name": "mlp-wide",
                    "primitive_family": "mlp",
                    "metric_name": "accuracy",
                    "metric_value": 0.94,
                    "status": "ok",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "Best Benchmarks" in result.output
    assert "moons" in result.output
    assert "mlp-wide" in result.output


def test_inspect_ignores_malformed_best_results_artifact_when_summary_omits_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "malformed_best_results_artifact"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "malformed_best_results_artifact",
                "runtime": "mlx",
                "evaluation_count": 1,
                "benchmark_count": 1,
                "failure_count": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_results.json").write_text("{not valid json", encoding="utf-8")

    result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "Best Benchmarks" not in result.output
    assert "Failure Count" in result.output


def test_inspect_handles_minimal_summary_without_optional_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "minimal_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "minimal_run",
                "runtime": "mlx",
                "evaluation_count": 1,
                "benchmark_count": 1,
                "failure_count": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "minimal_run" in result.output
    assert "Primitive Usage" not in result.output
    assert "Benchmark Group Coverage" not in result.output
    assert "Primitive Bank" not in result.output
    assert "Recent Failures" not in result.output
    assert "Best Benchmarks" not in result.output


def test_main_callback_reports_primitive_bank_export_ready() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Primitive bank export" in result.output
    assert "ready" in result.output
