import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli.main import app
from evonn_compare.output_quality import inspect_run_dir


runner = CliRunner()


def test_output_quality_writes_normalized_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "prism" / "case"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "system": "prism",
                "run_id": "case",
                "runtime_backend": "mlx",
                "runtime_backend_requested": "auto",
                "median_benchmark_quality": 0.9,
                "wall_clock_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "system": "prism",
                "run_id": "case",
                "run_name": "case",
                "created_at": "2026-05-05T00:00:00Z",
                "pack_name": "tier1_core_eval64",
                "seed": 42,
                "benchmarks": [
                    {
                        "benchmark_id": "iris",
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                        "status": "ok",
                    }
                ],
                "budget": {
                    "evaluation_count": 64,
                    "epochs_per_candidate": 20,
                    "wall_clock_seconds": 10.0,
                    "budget_policy_name": "prototype_equal_budget",
                    "actual_evaluations": 64,
                    "cached_evaluations": 0,
                    "failed_evaluations": 0,
                    "invalid_evaluations": 0,
                    "evaluation_semantics": "one candidate evaluation",
                },
                "device": {
                    "device_name": "apple_silicon",
                    "precision_mode": "fp32",
                    "framework": "mlx",
                    "framework_version": "0.31.1",
                },
                "artifacts": {
                    "config_snapshot": "config.yaml",
                    "report_markdown": "report.md",
                },
                "fairness": {
                    "benchmark_pack_id": "tier1_core_eval64",
                    "seed": 42,
                    "evaluation_count": 64,
                    "budget_policy_name": "prototype_equal_budget",
                    "data_signature": "abc",
                    "code_version": "deadbeef",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "system": "prism",
                    "run_id": "case",
                    "benchmark_id": "iris",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": 1.0,
                    "quality": 1.0,
                    "parameter_count": 10,
                    "train_seconds": 2.0,
                    "status": "ok",
                }
            ]
        ),
        encoding="utf-8",
    )

    record = inspect_run_dir(run_dir)

    assert record.quality_level == "L3"
    assert record.measurement_state == "measurable"
    assert record.performance.evals_per_second == 6.4
    assert (run_dir / "performance.json").exists()
    assert (run_dir / "diagnostics.json").exists()
    assert (run_dir / "output_quality_report.md").exists()


def test_output_quality_cli_accepts_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "runs" / "contenders" / "case"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "system": "contenders",
                "run_id": "case",
                "benchmarks": [],
                "budget": {"evaluation_count": 1},
                "device": {"device_name": "cpu"},
                "artifacts": {"config_snapshot": "config.yaml", "report_markdown": "report.md"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(app, ["output-quality", str(workspace)])

    assert result.exit_code == 0
    assert "runs\t1" in result.stdout
    assert (workspace / "output_quality_overview.md").exists()
