from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prism.runtime.training import _compute_metric, _metric_name


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_41bench.py"
    spec = importlib.util.spec_from_file_location("prism_smoke_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DummyPreprocessor:
    def fit_transform(self, x):
        return np.asarray(x)

    def transform(self, x):
        return np.asarray(x)


class _DummyBenchmark:
    id = "moons"
    modality = "tabular"

    @staticmethod
    def load_data(seed=42):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [0.8, 0.9]], dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int32)
        return X, y, X, y


class _DummyGenome:
    def __init__(self, family="mlp", hidden_layers=None, learning_rate=0.01):
        self.family = family
        self.hidden_layers = hidden_layers or [8, 4]
        self.learning_rate = learning_rate
        self.genome_id = f"{family}-g"

    def model_copy(self, update):
        clone = _DummyGenome(
            family=self.family,
            hidden_layers=list(self.hidden_layers),
            learning_rate=self.learning_rate,
        )
        for key, value in update.items():
            setattr(clone, key, value)
        return clone


def test_compute_metric_classification_returns_accuracy():
    metric_name, metric_value, quality = _compute_metric(
        "classification",
        np.array([0, 1, 1]),
        np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]]),
    )

    assert metric_name == "accuracy"
    assert metric_value == 1.0
    assert quality == 1.0


def test_compute_metric_regression_returns_mse_and_negative_quality():
    metric_name, metric_value, quality = _compute_metric(
        "regression",
        np.array([1.0, 3.0]),
        np.array([1.0, 2.0]),
    )

    assert metric_name == "mse"
    assert metric_value == 0.5
    assert quality == -0.5


def test_compute_metric_language_modeling_returns_perplexity_and_negative_quality():
    metric_name, metric_value, quality = _compute_metric(
        "language_modeling",
        np.array([[0, 1], [2, 0]], dtype=np.int64),
        np.array(
            [
                [[4.0, 1.0, 0.0], [0.0, 5.0, 1.0]],
                [[0.0, 1.0, 3.0], [2.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )

    assert metric_name == "perplexity"
    assert metric_value >= 1.0
    assert quality <= 0.0


def test_metric_name_helper_matches_task():
    assert _metric_name("classification") == "accuracy"
    assert _metric_name("regression") == "mse"


def test_mini_evolve_returns_canonical_contract(monkeypatch):
    smoke = _load_smoke_module()
    smoke.POP_SIZE = 1
    smoke.NUM_GENS = 1
    smoke.EPOCHS = 1

    monkeypatch.setattr(smoke, "get_benchmark", lambda name: _DummyBenchmark())
    monkeypatch.setattr(smoke, "Preprocessor", _DummyPreprocessor)
    monkeypatch.setattr(smoke, "compatible_families", lambda modality: ["mlp"])
    monkeypatch.setattr(
        smoke,
        "create_seed_genome",
        lambda fam, evolution, rng: _DummyGenome(family=fam),
    )
    monkeypatch.setattr(
        smoke,
        "compile_genome",
        lambda genome, input_shape, num_classes, modality: SimpleNamespace(
            model=object(),
            parameter_count=123,
        ),
    )
    monkeypatch.setattr(
        smoke,
        "train_and_evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            metric_name="accuracy",
            metric_value=0.91,
            quality=0.91,
            train_seconds=0.12,
            failure_reason=None,
        ),
    )

    outcome = smoke.mini_evolve("moons", "classification", seed=42)

    assert outcome["metric_name"] == "accuracy"
    assert outcome["metric_direction"] == "max"
    assert outcome["metric_value"] == 0.91
    assert outcome["quality"] == 0.91
    assert outcome["native_fitness"] == 0.91
    assert outcome["failure_reason"] is None
    assert outcome["genome"].family == "mlp"


def test_mini_evolve_returns_missing_contract_when_all_evals_fail(monkeypatch):
    smoke = _load_smoke_module()
    smoke.POP_SIZE = 1
    smoke.NUM_GENS = 1
    smoke.EPOCHS = 1

    monkeypatch.setattr(smoke, "get_benchmark", lambda name: _DummyBenchmark())
    monkeypatch.setattr(smoke, "Preprocessor", _DummyPreprocessor)
    monkeypatch.setattr(smoke, "compatible_families", lambda modality: ["mlp"])
    monkeypatch.setattr(
        smoke,
        "create_seed_genome",
        lambda fam, evolution, rng: _DummyGenome(family=fam),
    )
    monkeypatch.setattr(
        smoke,
        "compile_genome",
        lambda genome, input_shape, num_classes, modality: SimpleNamespace(
            model=object(),
            parameter_count=123,
        ),
    )
    monkeypatch.setattr(
        smoke,
        "train_and_evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    outcome = smoke.mini_evolve("moons", "classification", seed=42)

    assert outcome["metric_name"] == "accuracy"
    assert outcome["metric_direction"] == "max"
    assert outcome["metric_value"] is None
    assert outcome["quality"] is None
    assert outcome["native_fitness"] is None
    assert outcome["failure_reason"] == "no_valid_result"


def test_main_fails_fast_when_runtime_unavailable(monkeypatch, capsys):
    smoke = _load_smoke_module()

    monkeypatch.setattr(
        smoke,
        "_require_runtime_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeError("runtime unavailable")),
    )
    monkeypatch.setattr(smoke, "PACK_PATH", SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(smoke, "load_pack", lambda path: pytest.fail("load_pack should not run"))

    exit_code = smoke.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "runtime unavailable" in captured.out
