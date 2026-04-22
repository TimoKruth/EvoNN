import json
import yaml
from typer.testing import CliRunner

from stratograph.cli import app
from stratograph.config import BenchmarkPoolConfig, load_config
from stratograph.export import export_symbiosis_contract
from stratograph.export.report import load_report_context
from stratograph.pipeline import build_execution_ladder, run_evolution
from stratograph.storage import RunStore


def test_pipeline_and_export(repo_root, tmp_path) -> None:
    base_config = load_config(repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml")
    config = base_config.model_copy(
        update={
            "run_name": "mini_export_pack",
            "benchmark_pool": BenchmarkPoolConfig(name="mini_export_pack", benchmarks=["moons", "tiny_lm_synthetic"]),
            "evolution": base_config.evolution.model_copy(update={"population_size": 2, "generations": 1}),
        }
    )
    config_path = tmp_path / "mini_export_config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False), encoding="utf-8")
    pack_path = tmp_path / "mini_export_pack.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "name": "mini_export_pack",
                "benchmarks": [
                    {
                        "benchmark_id": "moons_classification",
                        "native_ids": {"stratograph": "moons"},
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                    },
                    {
                        "benchmark_id": "tiny_lm_synthetic",
                        "native_ids": {"stratograph": "tiny_lm_synthetic"},
                        "task_kind": "language_modeling",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                    },
                ],
                "budget_policy": {
                    "evaluation_count": 2,
                    "epochs_per_candidate": 1,
                    "budget_tolerance_pct": 10.0,
                },
                "seed_policy": {"mode": "shared", "required": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "prototype_run"
    run_evolution(config, run_dir=run_dir, config_path=config_path)

    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    results = store.load_results(run_dir.name)
    budget_meta = store.load_budget_metadata(run_dir.name)
    store.close()

    assert len(runs) == 1
    assert len(results) == 2
    assert {record["status"] for record in results} <= {"ok", "failed"}
    assert budget_meta["runtime_backend"] in {"mlx", "numpy-fallback"}
    assert "runtime_version" in budget_meta
    assert budget_meta["precision_mode"] == "fp32"

    manifest_path, results_path = export_symbiosis_contract(
        run_dir,
        pack_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exported_results = json.loads(results_path.read_text(encoding="utf-8"))

    assert manifest["system"] == "stratograph"
    assert manifest["pack_name"] == "mini_export_pack"
    assert len(manifest["benchmarks"]) == 2
    assert len(exported_results) == 2
    assert manifest["artifacts"]["config_snapshot"] == "config.yaml"
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert manifest["fairness"]["evaluation_count"] == manifest["budget"]["evaluation_count"]
    assert manifest["device"]["framework"] == budget_meta["runtime_backend"]
    assert manifest["device"]["framework_version"] == (budget_meta["runtime_version"] or "unknown")
    assert manifest["device"]["precision_mode"] == budget_meta["precision_mode"]

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert f"- Runtime: `{budget_meta['runtime_backend']}`" in report
    expected_version = budget_meta["runtime_version"] or "unknown"
    assert f"- Runtime Version: `{expected_version}`" in report
    assert f"- Precision Mode: `{budget_meta['precision_mode']}`" in report
    assert f"- Created At: `{budget_meta['created_at']}`" in report
    assert f"- Run State: `{status['state']}`" in report
    assert f"- Completed Benchmarks: `{status['completed_count']}/{status['total_benchmarks']}`" in report
    assert f"- Remaining Benchmarks: `{status['remaining_count']}`" in report
    assert "- Status Artifact: `status.json`" in report
    assert "- Checkpoint Artifact: `checkpoint.json`" in report
    assert f"- Effective Training Epochs: `{budget_meta['effective_training_epochs']}`" in report
    assert f"- Architecture Mode: `{budget_meta['architecture_mode']}`" in report
    assert "## Benchmarks" in report
    assert "## Best Benchmarks" in report
    assert "## Failure Details" in report
    assert "| Benchmark | Metric | Value | Quality | Params | Train Seconds | Genome | Architecture |" in report
    assert "| Benchmark | Reason |" in report


def test_inspect_command_surfaces_rich_run_summary(repo_root, tmp_path) -> None:
    runner = CliRunner()
    config_path = repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml"
    base_config = load_config(config_path)
    config = base_config.model_copy(
        update={
            "run_name": "inspect_summary",
            "benchmark_pool": BenchmarkPoolConfig(name="inspect_summary", benchmarks=["moons", "tiny_lm_synthetic"]),
            "evolution": base_config.evolution.model_copy(update={"population_size": 2, "generations": 1}),
        }
    )
    run_dir = tmp_path / "inspect_summary"
    run_evolution(config, run_dir=run_dir, config_path=config_path)

    result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert "Run Overview" in result.stdout
    assert "Best Benchmarks" in result.stdout
    assert "Failure Details" in result.stdout
    assert "Created At" in result.stdout
    assert "Run State" in result.stdout
    assert "Runtime Version" in result.stdout
    assert "Precision Mode" in result.stdout
    assert "Completed Benchmarks" in result.stdout
    assert "Remaining Benchmarks" in result.stdout
    assert "Status Artifact" in result.stdout
    assert "Occupied Niches" in result.stdout


def test_inspect_command_handles_empty_run_dir(tmp_path) -> None:
    runner = CliRunner()
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    result = runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])

    assert result.exit_code == 1


def test_report_context_keeps_best_result_per_benchmark(tmp_path) -> None:
    run_dir = tmp_path / "dedupe_run"
    with RunStore(run_dir / "metrics.duckdb") as store:
        store.record_run(
            run_id="dedupe_run",
            run_name="dedupe_run",
            created_at="2026-04-22T00:00:00",
            seed=7,
            config={},
        )
        store.save_budget_metadata(run_id="dedupe_run", payload={"runtime_backend": "numpy-fallback"})
        store.record_result(
            run_id="dedupe_run",
            benchmark_name="moons",
            record={
                "metric_name": "accuracy",
                "metric_direction": "max",
                "metric_value": 0.75,
                "quality": 0.75,
                "parameter_count": 100,
                "train_seconds": 1.0,
                "architecture_summary": "weaker",
                "genome_id": "g1",
                "status": "ok",
                "failure_reason": None,
            },
        )
        store.record_result(
            run_id="dedupe_run",
            benchmark_name="moons",
            record={
                "metric_name": "accuracy",
                "metric_direction": "max",
                "metric_value": 0.9,
                "quality": 0.9,
                "parameter_count": 120,
                "train_seconds": 1.2,
                "architecture_summary": "stronger",
                "genome_id": "g2",
                "status": "ok",
                "failure_reason": None,
            },
        )

    context = load_report_context(run_dir)

    assert len(context["best_results"]) == 1
    assert context["best_results"][0]["genome_id"] == "g2"


def test_build_execution_ladder(tmp_path) -> None:
    cases = build_execution_ladder(tmp_path / "ladder")
    assert len(cases) == 9
    assert cases[0].name == "single_moons_classification"
    assert cases[-1].name.endswith("eval608")
