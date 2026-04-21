import json
import yaml

from stratograph.config import BenchmarkPoolConfig, load_config
from stratograph.export import export_symbiosis_contract
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
    assert manifest["device"]["framework_version"] == budget_meta["runtime_version"]

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert f"- Runtime: `{budget_meta['runtime_backend']}`" in report
    expected_version = budget_meta["runtime_version"] or "unknown"
    assert f"- Runtime Version: `{expected_version}`" in report
    assert f"- Effective Training Epochs: `{budget_meta['effective_training_epochs']}`" in report
    assert f"- Architecture Mode: `{budget_meta['architecture_mode']}`" in report
    assert "## Benchmarks" in report


def test_build_execution_ladder(tmp_path) -> None:
    cases = build_execution_ladder(tmp_path / "ladder")
    assert len(cases) == 9
    assert cases[0].name == "single_moons_classification"
    assert cases[-1].name.endswith("eval608")
