from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evonn_primordia.config import RunConfig
from evonn_primordia.pipeline import run_search
from evonn_primordia import pipeline as primordia_pipeline
from evonn_primordia.runtime.backends import _build_classification_estimator, resolve_runtime_bindings


def test_resolve_runtime_bindings_respects_explicit_numpy_fallback() -> None:
    config = RunConfig.model_validate(
        {
            "benchmark_pool": {"name": "mini", "benchmarks": ["iris"]},
            "runtime": {"backend": "numpy-fallback"},
        }
    )

    runtime = resolve_runtime_bindings(config)

    assert runtime.runtime_backend == "numpy-fallback"
    assert runtime.runtime_version.startswith("sklearn-")
    assert runtime.precision_mode == "fp32"


def test_numpy_fallback_run_search_executes_real_metrics(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: fallback_smoke
runtime:
  backend: numpy-fallback
benchmark_pool:
  name: fallback_smoke
  benchmarks:
    - iris
    - diabetes
search:
  mode: budget_matched
  target_evaluation_count: 4
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp, sparse_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = RunConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    run_dir = tmp_path / "run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trials = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))

    assert summary["runtime"] == "numpy-fallback"
    assert summary["runtime_optimizations"]["benchmark_preprocessing"] == "benchmark_cached_scaler"
    assert summary["runtime_optimizations"]["candidate_batch_execution"] == "disabled"
    assert summary["failure_count"] < summary["evaluation_count"]
    assert any(record["status"] == "ok" and record["metric_value"] is not None for record in trials)
    assert all(record["runtime"] == "numpy-fallback" for record in trials)


def test_numpy_fallback_handles_repeated_slots_without_tuple_crash(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: repeated_slots
runtime:
  backend: numpy-fallback
benchmark_pool:
  name: repeated_slots
  benchmarks:
    - iris
search:
  mode: budget_matched
  target_evaluation_count: 5
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp, sparse_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = RunConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    run_dir = tmp_path / "repeated_slots_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trials = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))

    assert summary["evaluation_count"] == 5
    assert len(trials) == 5
    assert any(record["slot_index"] >= 2 for record in trials)
    assert any(record["status"] == "ok" for record in trials)


def test_numpy_fallback_preprocesses_once_per_benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 42
run_name: cached_preprocessing
runtime:
  backend: numpy-fallback
benchmark_pool:
  name: cached_preprocessing
  benchmarks:
    - iris
search:
  mode: budget_matched
  target_evaluation_count: 5
training:
  epochs_per_candidate: 1
primitive_pool:
  tabular: [mlp, sparse_mlp]
  synthetic: [mlp]
  image: [mlp]
  language_modeling: [embedding]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    fit_calls = 0
    transform_calls = 0
    fit_transform = primordia_pipeline.StandardScaler.fit_transform
    transform = primordia_pipeline.StandardScaler.transform

    def counted_fit_transform(self, values, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return fit_transform(self, values, *args, **kwargs)

    def counted_transform(self, values, *args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return transform(self, values, *args, **kwargs)

    monkeypatch.setattr(primordia_pipeline.StandardScaler, "fit_transform", counted_fit_transform)
    monkeypatch.setattr(primordia_pipeline.StandardScaler, "transform", counted_transform)

    config = RunConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    run_dir = tmp_path / "cached_preprocessing_run"
    run_search(config, run_dir=run_dir, config_path=config_path)

    assert fit_calls == 1
    assert transform_calls == 2


def test_numpy_fallback_conv_estimators_avoid_removed_multi_class_kwarg() -> None:
    genome = type("Genome", (), {"hidden_layers": [64, 64], "activation": "relu"})()

    for family in ("conv2d", "lite_conv2d"):
        estimator = _build_classification_estimator(family, genome, epochs=1, lr=1e-3, weight_decay=0.0)
        params = estimator.get_params()
        assert "multi_class" not in params or params["multi_class"] != "auto"
