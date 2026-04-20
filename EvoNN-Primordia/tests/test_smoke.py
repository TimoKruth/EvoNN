from pathlib import Path
import json

from evonn_primordia.config import load_config
from evonn_primordia.export.symbiosis import export_symbiosis_contract
from evonn_primordia.pipeline import run_search


def test_smoke_run_and_export(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: smoke_primordia
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
  - tiny_lm_synthetic
search:
  mode: budget_matched
  target_evaluation_count: 4
primitive_pool:
  tabular: [logistic, mlp_small]
  synthetic: [linear_svc, mlp]
  image: [logistic, mlp_small]
  language_modeling: [unigram_lm, bigram_lm]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        """
name: smoke_pack_eval4
benchmarks:
  - benchmark_id: iris_classification
    native_ids:
      primordia: iris
    task_kind: classification
    metric_name: accuracy
    metric_direction: max
  - benchmark_id: tiny_lm_synthetic
    native_ids:
      primordia: tiny_lm_synthetic
    task_kind: language_modeling
    metric_name: perplexity
    metric_direction: min
budget_policy:
  evaluation_count: 4
  epochs_per_candidate: 1
seed_policy:
  mode: campaign
  required: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path, results_path = export_symbiosis_contract(run_dir, pack_path, run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert manifest["system"] == "primordia"
    assert manifest["fairness"]["benchmark_pack_id"] == manifest["pack_name"]
    assert manifest["fairness"]["evaluation_count"] == manifest["budget"]["evaluation_count"]
    benchmark_ids = {record["benchmark_id"] for record in results}
    assert "iris_classification" in benchmark_ids
    assert "tiny_lm_synthetic" in benchmark_ids


def test_budget_matched_mode_hits_exact_target_evaluation_count(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: budgeted_primordia
benchmark_pool:
  name: smoke_pack
  benchmarks:
  - iris
  - circles
search:
  mode: budget_matched
  target_evaluation_count: 6
primitive_pool:
  tabular: [logistic, mlp_small]
  synthetic: [linear_svc, mlp]
  image: [logistic]
  language_modeling: [unigram_lm]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trials = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))
    assert summary["evaluation_count"] == 6
    assert len(trials) == 6
    assert {record["benchmark_name"] for record in trials} == {"iris", "circles"}
    assert any(record["primitive_name"].endswith("@r2") for record in trials)
