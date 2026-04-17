from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from topograph.export.symbiosis import _benchmark_metric_direction, _benchmark_metric_name
from topograph.nn import train as train_mod


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_41bench.py"
    spec = importlib.util.spec_from_file_location("topograph_smoke_test_module", path)
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
    @staticmethod
    def load_data(seed=42):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [0.8, 0.9]], dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int32)
        return X, y, X, y


class _DummyInnovationCounter:
    pass


class _DummyGenome:
    def __init__(self):
        self.layers = [1, 2, 3]
        self.connections = [1, 2]
        self.enabled_layers = [1, 2, 3]
        self.enabled_connections = [1, 2]
        self.learning_rate = 0.01
        self.batch_size = 32
        self.fitness = None
        self.model_bytes = None

    @classmethod
    def create_seed(cls, ic, rng, num_layers=3):
        return cls()


def test_compute_metric_classification_returns_accuracy():
    metric_name, metric_direction, metric_value, quality = train_mod._compute_metric(
        "classification",
        np.array([0, 1, 1]),
        np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]]),
    )

    assert metric_name == "accuracy"
    assert metric_direction == "max"
    assert metric_value == 1.0
    assert quality == 1.0


def test_compute_metric_regression_returns_mse_and_negative_quality():
    metric_name, metric_direction, metric_value, quality = train_mod._compute_metric(
        "regression",
        np.array([1.0, 3.0]),
        np.array([1.0, 2.0]),
    )

    assert metric_name == "mse"
    assert metric_direction == "min"
    assert metric_value == 0.5
    assert quality == -0.5


def test_train_model_returns_native_fitness_from_wrapper(monkeypatch):
    monkeypatch.setattr(
        train_mod,
        "train_and_evaluate",
        lambda *args, **kwargs: SimpleNamespace(native_fitness=0.37),
    )

    native_fitness = train_mod.train_model(
        model=None,
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        epochs=1,
        lr=0.01,
        batch_size=32,
    )

    assert native_fitness == 0.37


def test_export_metric_labels_are_cross_system_aligned():
    assert _benchmark_metric_name("classification") == "accuracy"
    assert _benchmark_metric_direction("classification") == "max"
    assert _benchmark_metric_name("regression") == "mse"
    assert _benchmark_metric_direction("regression") == "min"


def test_mini_evolve_returns_canonical_contract(monkeypatch):
    smoke = _load_smoke_module()
    smoke.POP_SIZE = 1
    smoke.NUM_GENS = 1
    smoke.EPOCHS = 1

    monkeypatch.setattr(smoke, "get_benchmark", lambda name: _DummyBenchmark())
    monkeypatch.setattr(smoke, "Preprocessor", _DummyPreprocessor)
    monkeypatch.setattr(smoke, "InnovationCounter", _DummyInnovationCounter)
    monkeypatch.setattr(smoke, "Genome", _DummyGenome)
    monkeypatch.setattr(smoke, "compile_genome", lambda *args, **kwargs: object())
    monkeypatch.setattr(smoke, "estimate_model_bytes", lambda genome: 256)
    monkeypatch.setattr(
        smoke,
        "train_and_evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            metric_name="accuracy",
            metric_direction="max",
            metric_value=0.88,
            quality=0.88,
            native_fitness=0.47,
            train_seconds=0.21,
            failure_reason=None,
        ),
    )

    outcome = smoke.mini_evolve("moons", "classification", seed=42)

    assert outcome["metric_name"] == "accuracy"
    assert outcome["metric_direction"] == "max"
    assert outcome["metric_value"] == 0.88
    assert outcome["quality"] == 0.88
    assert outcome["native_fitness"] == 0.47
    assert outcome["failure_reason"] is None


def test_mini_evolve_returns_missing_contract_when_all_evals_fail(monkeypatch):
    smoke = _load_smoke_module()
    smoke.POP_SIZE = 1
    smoke.NUM_GENS = 1
    smoke.EPOCHS = 1

    monkeypatch.setattr(smoke, "get_benchmark", lambda name: _DummyBenchmark())
    monkeypatch.setattr(smoke, "Preprocessor", _DummyPreprocessor)
    monkeypatch.setattr(smoke, "InnovationCounter", _DummyInnovationCounter)
    monkeypatch.setattr(smoke, "Genome", _DummyGenome)
    monkeypatch.setattr(smoke, "compile_genome", lambda *args, **kwargs: object())
    monkeypatch.setattr(smoke, "estimate_model_bytes", lambda genome: 256)
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
