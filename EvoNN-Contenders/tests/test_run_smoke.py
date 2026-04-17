from pathlib import Path

import json

from evonn_contenders.config import load_config
from evonn_contenders.export.symbiosis import export_symbiosis_contract
from evonn_contenders.pipeline import run_contenders


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
    benchmark_ids = {record["benchmark_id"] for record in results}
    assert "iris_classification" in benchmark_ids
    assert "tiny_lm_synthetic" in benchmark_ids
