from pathlib import Path

import json

from evonn_contenders.config import load_config
from evonn_contenders.export.symbiosis import export_symbiosis_contract
from evonn_contenders.pipeline import materialize_baseline_run, run_contenders
from evonn_contenders.storage import RunStore


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
    run_dir = tmp_path / "run"
    config = load_config(config_path)
    run_contenders(config, run_dir=run_dir, config_path=config_path)
    manifest_path, results_path = export_symbiosis_contract(
        run_dir,
        Path("/Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Compare/manual_compare_runs/20260416_budget38_seed42_smoke_valid/packs/working_33_plus_5_lm_compare_eval38.yaml"),
        run_dir,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert manifest["system"] == "contenders"
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert manifest["fairness"]["evaluation_count"] == manifest["budget"]["evaluation_count"]
    benchmark_ids = {record["benchmark_id"] for record in results}
    assert "iris_classification" in benchmark_ids
    assert "tiny_lm_synthetic" in benchmark_ids


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
