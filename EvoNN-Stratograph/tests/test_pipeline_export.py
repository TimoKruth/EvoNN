import json

from stratograph.config import load_config
from stratograph.export import export_symbiosis_contract
from stratograph.pipeline import build_execution_ladder, run_evolution
from stratograph.storage import RunStore


def test_pipeline_and_export(repo_root, tmp_path) -> None:
    config_path = repo_root / "configs" / "working_33_plus_5_lm_smoke.yaml"
    config = load_config(config_path)
    run_dir = tmp_path / "prototype_run"
    run_evolution(config, run_dir=run_dir, config_path=config_path)

    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    results = store.load_results(run_dir.name)
    store.close()

    assert len(runs) == 1
    assert len(results) == 38
    assert {record["status"] for record in results} <= {"ok", "failed"}

    manifest_path, results_path = export_symbiosis_contract(
        run_dir,
        config_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exported_results = json.loads(results_path.read_text(encoding="utf-8"))

    assert manifest["system"] == "stratograph"
    assert manifest["pack_name"] == "working_33_plus_5_lm_smoke"
    assert len(manifest["benchmarks"]) == 38
    assert len(exported_results) == 38
    assert manifest["artifacts"]["config_snapshot"] == "config.yaml"
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert manifest["fairness"]["evaluation_count"] == manifest["budget"]["evaluation_count"]


def test_build_execution_ladder(tmp_path) -> None:
    cases = build_execution_ladder(tmp_path / "ladder")
    assert len(cases) == 9
    assert cases[0].name == "single_moons_classification"
    assert cases[-1].name.endswith("eval608")
