from pathlib import Path

from evonn_compare.hybrid.benchmarks import load_lm_benchmarks, load_parity_pack_benchmarks


def test_load_synthetic_lm_benchmark() -> None:
    benchmarks = load_lm_benchmarks(seed=42)
    assert "tiny_lm_synthetic" in benchmarks
    x_train, y_train, x_val, y_val, task, vocab_size = benchmarks["tiny_lm_synthetic"]
    assert task == "language_modeling"
    assert vocab_size == 256
    assert x_train.ndim == 2
    assert y_train.ndim == 1
    assert x_val.ndim == 2
    assert y_val.ndim == 2


def test_load_lm_pack_benchmarks(tmp_path: Path) -> None:
    pack_path = tmp_path / "lm_pack.yaml"
    pack_path.write_text(
        """
name: lm_bridge
tier: 3
description: test lm pack
benchmarks:
  - benchmark_id: tiny_lm_synthetic
    native_ids:
      evonn: tiny_lm_synthetic
      hybrid: tiny_lm_synthetic
    task_kind: language_modeling
    metric_name: perplexity
    metric_direction: min
budget_policy:
  evaluation_count: 4
  epochs_per_candidate: 1
  budget_tolerance_pct: 10.0
seed_policy:
  mode: shared
  required: true
""".strip(),
        encoding="utf-8",
    )

    benchmarks = load_parity_pack_benchmarks(pack_path, seed=42)
    assert "tiny_lm_synthetic" in benchmarks
    _, _, _, y_val, task, vocab_size = benchmarks["tiny_lm_synthetic"]
    assert task == "language_modeling"
    assert vocab_size == 256
    assert y_val.ndim == 2
