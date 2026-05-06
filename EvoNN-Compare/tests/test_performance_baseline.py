import json
from pathlib import Path

from typer.testing import CliRunner

from evonn_compare.cli.main import app
from evonn_compare.orchestration.performance_baseline import build_performance_baseline


runner = CliRunner()


def _write_run(run_dir: Path, *, system: str, run_id: str, budget: int, wall_clock: float, quality: float = 0.9) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "system": system,
                "run_id": run_id,
                "runtime_backend": "mlx",
                "runtime_backend_requested": "auto",
                "wall_clock_seconds": wall_clock,
                "median_benchmark_quality": quality,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "system": system,
                "run_id": run_id,
                "run_name": run_id,
                "created_at": "2026-05-06T00:00:00Z",
                "pack_name": f"tier1_core_eval{budget}",
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
                    "evaluation_count": budget,
                    "epochs_per_candidate": 20,
                    "wall_clock_seconds": wall_clock,
                    "budget_policy_name": "prototype_equal_budget",
                    "actual_evaluations": budget,
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
                    "benchmark_pack_id": f"tier1_core_eval{budget}",
                    "seed": 42,
                    "evaluation_count": budget,
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
                    "system": system,
                    "run_id": run_id,
                    "benchmark_id": "iris",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": quality,
                    "quality": quality,
                    "parameter_count": 10,
                    "train_seconds": 2.0,
                    "status": "ok",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_build_performance_baseline_writes_bundle(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root / "prism64", system="prism", run_id="prism64", budget=64, wall_clock=10.0)
    _write_run(runs_root / "prism256", system="prism", run_id="prism256", budget=256, wall_clock=20.0)
    _write_run(runs_root / "prism1000", system="prism", run_id="prism1000", budget=1000, wall_clock=50.0)
    _write_run(runs_root / "contenders64", system="contenders", run_id="contenders64", budget=64, wall_clock=12.0)

    result = build_performance_baseline(inputs=[runs_root], output_root=tmp_path / "baselines")

    payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    prism = next(row for row in payload["systems"] if row["system"] == "prism")
    contenders = next(row for row in payload["systems"] if row["system"] == "contenders")

    assert result["accepted_run_count"] == 4
    assert prism["performance_claim_ready"] is True
    assert prism["budgets_present"] == [64, 256, 1000]
    assert contenders["performance_claim_ready"] is False
    assert contenders["missing_budgets"] == [256, 1000]
    assert Path(result["markdown"]).exists()
    assert Path(result["jsonl"]).exists()


def test_performance_baseline_cli_accepts_workspace(tmp_path: Path) -> None:
    runs_root = tmp_path / "workspace"
    _write_run(runs_root / "runs" / "prism64", system="prism", run_id="prism64", budget=64, wall_clock=10.0)

    result = runner.invoke(app, ["performance-baseline", str(runs_root), "--output-root", str(tmp_path / "perf")])

    assert result.exit_code == 0
    assert "bundle_root\t" in result.stdout
    assert "accepted_runs\t1" in result.stdout
