import importlib.util

import pytest

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.config import RunConfig
from evonn_contenders.contender_pool import evaluate_contender, resolve_contenders
from evonn_contenders.contenders.torch_models import build_cnn


def _smoke_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "seed": 7,
            "benchmark_pool": {"name": "smoke", "benchmarks": ["iris"]},
            "contender_pool": {
                "tabular": ["linear_svc"],
                "synthetic": ["linear_svc"],
                "image": ["cnn_small"],
                "language_modeling": ["transformer_lm_tiny"],
            },
            "torch": {
                "batch_size": 32,
                "classifier_epochs": 1,
                "max_batches_per_epoch": 4,
                "lm_steps": 4,
                "max_train_samples": 64,
                "max_val_samples": 32,
                "context_length_override": 32,
            },
        }
    )


def _evaluate(benchmark_name: str, group: str, contender_name: str):
    spec = get_benchmark(benchmark_name)
    contender = resolve_contenders(group, [contender_name])[0]
    x_train, y_train, x_val, y_val = spec.load_data(seed=7)
    return evaluate_contender(
        spec,
        contender,
        seed=7,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        config=_smoke_config(),
    )


def test_linear_svc_trains_on_iris() -> None:
    record = _evaluate("iris", "tabular", "linear_svc")
    assert record["status"] == "ok"
    assert 0.0 <= float(record["metric_value"]) <= 1.0


def test_diabetes_benchmark_loads_regression_data() -> None:
    spec = get_benchmark("diabetes")
    x_train, y_train, x_val, y_val = spec.load_data(seed=7)
    assert x_train.shape[1] == 10
    assert y_train.dtype.kind == "f"
    assert y_val.dtype.kind == "f"


def test_friedman1_benchmark_loads_regression_data() -> None:
    spec = get_benchmark("friedman1")
    x_train, y_train, x_val, y_val = spec.load_data(seed=7)
    assert x_train.shape[1] == 10
    assert y_train.dtype.kind == "f"
    assert y_val.dtype.kind == "f"


def test_hist_gb_trains_on_diabetes_regression() -> None:
    record = _evaluate("diabetes", "tabular", "hist_gb")
    assert record["status"] == "ok"
    assert float(record["metric_value"]) >= 0.0


def test_extra_trees_trains_on_friedman1_regression() -> None:
    record = _evaluate("friedman1", "tabular", "extra_trees")
    assert record["status"] == "ok"
    assert float(record["metric_value"]) >= 0.0


def test_new_contender_names_resolve() -> None:
    assert resolve_contenders("tabular", ["linear_svc", "xgb_small", "lgbm_small", "catboost_small"])
    assert resolve_contenders("image", ["cnn_small"])
    assert resolve_contenders("language_modeling", ["transformer_lm_tiny"])


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost not installed")
def test_xgboost_trains_on_iris() -> None:
    record = _evaluate("iris", "tabular", "xgb_small")
    assert record["status"] == "ok"
    assert 0.0 <= float(record["metric_value"]) <= 1.0


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_lightgbm_trains_on_iris() -> None:
    record = _evaluate("iris", "tabular", "lgbm_small")
    assert record["status"] == "ok"
    assert 0.0 <= float(record["metric_value"]) <= 1.0


@pytest.mark.skipif(importlib.util.find_spec("catboost") is None, reason="catboost not installed")
def test_catboost_trains_on_iris() -> None:
    record = _evaluate("iris", "tabular", "catboost_small")
    assert record["status"] == "ok"
    assert 0.0 <= float(record["metric_value"]) <= 1.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_cnn_trains_on_digits() -> None:
    record = _evaluate("digits", "image", "cnn_small")
    assert record["status"] == "ok"
    assert 0.0 <= float(record["metric_value"]) <= 1.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_cnn_avoids_maxpool_deadlock_path() -> None:
    import torch.nn as nn

    model = build_cnn("cnn_small", channels=1, num_classes=10)
    assert not any(isinstance(layer, nn.MaxPool2d) for layer in model)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_transformer_trains_on_tiny_lm_synthetic() -> None:
    record = _evaluate("tiny_lm_synthetic", "language_modeling", "transformer_lm_tiny")
    assert record["status"] == "ok"
    assert float(record["metric_value"]) > 0.0
