from pathlib import Path

import json
from typer.testing import CliRunner

from evonn_contenders.cli import app
from evonn_contenders.config import load_config
from evonn_contenders.export.symbiosis import export_symbiosis_contract
from evonn_contenders.pipeline import materialize_baseline_run, run_contenders
from evonn_contenders.storage import RunStore

runner = CliRunner()


def test_materialize_cli_reports_cache_miss_without_traceback(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: cached_contenders
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
baseline:
  baseline_id: smoke_fixed
  cache_dir: baseline-cache
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fail_materialize(*args, **kwargs):
        raise ValueError("benchmark 'iris' missing from baseline cache 'smoke_fixed'")

    monkeypatch.setattr("evonn_contenders.cli.materialize_baseline_run", fail_materialize)

    result = runner.invoke(app, ["materialize", "--config", str(config_path), "--run-dir", str(tmp_path / "run")])

    assert result.exit_code == 1
    assert "missing from baseline cache" in result.output
    assert "Traceback" not in result.output


def test_smoke_run_and_export(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: smoke_contenders
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
  - tiny_lm_synthetic
contender_pool:
  tabular: [logistic, hist_gb]
  synthetic: [hist_gb, extra_trees]
  image: [mlp]
  language_modeling: [bigram_lm, unigram_lm]
selection:
  max_contenders_per_benchmark: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    pack_path = tmp_path / "smoke_compare_pack.yaml"
    pack_path.write_text(
        """
name: smoke_compare_pack
benchmarks:
  - benchmark_id: iris_classification
    native_ids:
      contenders: iris
    task_kind: classification
    metric_name: accuracy
    metric_direction: max
  - benchmark_id: tiny_lm_synthetic
    native_ids:
      contenders: tiny_lm_synthetic
    task_kind: language_modeling
    metric_name: perplexity
    metric_direction: min
budget_policy:
  evaluation_count: 4
  epochs_per_candidate: 1
seed_policy:
  mode: shared
  required: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    config = load_config(config_path)
    run_contenders(config, run_dir=run_dir, config_path=config_path)
    manifest_path, results_path = export_symbiosis_contract(run_dir, pack_path, run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert manifest["system"] == "contenders"
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert manifest["fairness"]["evaluation_count"] == manifest["budget"]["evaluation_count"]
    benchmark_ids = {record["benchmark_id"] for record in results}
    assert "iris_classification" in benchmark_ids
    assert "tiny_lm_synthetic" in benchmark_ids
    assert summary["system"] == "contenders"
    assert summary["run_id"] == manifest["run_id"]
    assert summary["benchmarks_evaluated"] == len(results)


def test_baseline_cache_reuses_existing_benchmarks(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: cached_contenders
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
baseline:
  baseline_id: smoke_fixed
  cache_dir: baseline-cache
contender_pool:
  tabular: [logistic, hist_gb]
  synthetic: [hist_gb]
  image: [mlp]
  language_modeling: [bigram_lm]
selection:
  max_contenders_per_benchmark: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    first_run = tmp_path / "run_a"
    run_contenders(config, run_dir=first_run, config_path=config_path)

    def fail_evaluate(*args, **kwargs):
        raise AssertionError("cache miss: contender evaluation should not rerun")

    monkeypatch.setattr("evonn_contenders.pipeline.evaluate_contender", fail_evaluate)
    second_run = tmp_path / "run_b"
    run_contenders(config, run_dir=second_run, config_path=config_path)

    store = RunStore(second_run / "metrics.duckdb")
    runs = store.load_runs()
    run = next(item for item in runs if item["run_id"] == "run_b")
    meta = store.load_budget_metadata(run["run_id"])
    results = store.load_results(run["run_id"])
    store.close()

    assert meta["cache_hits"] == 1
    assert meta["executed_evaluation_count"] == 0
    assert len(results) == 1


def test_materialize_baseline_run_uses_cached_results(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: cached_contenders
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
baseline:
  baseline_id: smoke_fixed
  cache_dir: baseline-cache
contender_pool:
  tabular: [logistic, hist_gb]
  synthetic: [hist_gb]
  image: [mlp]
  language_modeling: [bigram_lm]
selection:
  max_contenders_per_benchmark: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    seed_run = tmp_path / "run_seed"
    run_contenders(config, run_dir=seed_run, config_path=config_path)

    materialized = tmp_path / "run_materialized"
    materialize_baseline_run(config, run_dir=materialized, config_path=config_path)
    store = RunStore(materialized / "metrics.duckdb")
    meta = store.load_budget_metadata("run_materialized")
    results = store.load_results("run_materialized")
    store.close()

    assert meta["cache_hits"] == 1
    assert meta["executed_evaluation_count"] == 0
    assert len(results) == 1


def test_budget_matched_mode_hits_exact_target_evaluation_count(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: budget_matched_contenders
benchmark_pool:
  name: tiny_pack
  benchmarks:
  - iris
  - circles
baseline:
  mode: budget_matched
  target_evaluation_count: 6
  cache_dir: baseline-cache
contender_pool:
  tabular: [logistic, hist_gb]
  synthetic: [hist_gb, extra_trees]
  image: [mlp]
  language_modeling: [bigram_lm]
selection:
  max_contenders_per_benchmark: null
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_contenders(config, run_dir=run_dir, config_path=config_path)

    store = RunStore(run_dir / "metrics.duckdb")
    meta = store.load_budget_metadata("run")
    contenders = store.load_contenders("run")
    store.close()

    assert meta["evaluation_count"] == 6
    assert meta["executed_evaluation_count"] == 6
    assert len(contenders) == 6
    assert {record["benchmark_name"] for record in contenders} == {"iris", "circles"}
    assert any(record["contender_name"].endswith("@r2") for record in contenders)


def test_optional_missing_contenders_are_skipped_by_default(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: optional_skip
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
selection:
  max_contenders_per_benchmark: null
""".strip()
        + "\n",
        encoding="utf-8",
    )
    real_find_spec = __import__("importlib.util").util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"xgboost", "lightgbm", "catboost"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("evonn_contenders.pipeline.importlib.util.find_spec", fake_find_spec)
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_contenders(config, run_dir=run_dir, config_path=config_path)

    store = RunStore(run_dir / "metrics.duckdb")
    contenders = store.load_contenders("run")
    results = store.load_results("run")
    store.close()

    contender_names = {record["contender_name"] for record in contenders}
    assert "xgb_small" not in contender_names
    assert "lgbm_small" not in contender_names
    assert "catboost_small" not in contender_names
    assert all(record["status"] == "ok" for record in contenders)
    assert results[0]["status"] == "ok"


def test_run_contenders_emits_progress_lines(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: progress_smoke
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
contender_pool:
  tabular: [logistic, hist_gb]
  synthetic: [hist_gb]
  image: [mlp]
  language_modeling: [bigram_lm]
selection:
  max_contenders_per_benchmark: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_contenders(config, run_dir=tmp_path / "run", config_path=config_path)

    output = capsys.readouterr().out
    assert "[evonn-contenders] start run_id=run" in output
    assert "[evonn-contenders] [1/1] benchmark=iris" in output
    assert "[evonn-contenders] finished run_id=run" in output
