from __future__ import annotations

import sys
import types

import numpy as np

from evonn_compare.shared_benchmarks import SharedBenchmarkSpec, _load_sklearn


def test_load_sklearn_preserves_zero_circle_factor(monkeypatch) -> None:
    calls: dict[str, float] = {}

    def make_circles(*, n_samples: int, noise: float, factor: float, random_state: int):
        calls["factor"] = factor
        return np.zeros((n_samples, 2), dtype=np.float32), np.zeros(n_samples, dtype=np.int64)

    sklearn = types.ModuleType("sklearn")
    datasets = types.ModuleType("sklearn.datasets")
    datasets.make_circles = make_circles
    sklearn.datasets = datasets
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.datasets", datasets)

    spec = SharedBenchmarkSpec(
        name="circles_zero_factor",
        task="classification",
        source="sklearn",
        dataset="make_circles",
        factor=0.0,
        n_samples=8,
    )

    _load_sklearn(spec, seed=42)

    assert calls["factor"] == 0.0


def test_load_sklearn_preserves_zero_blob_cluster_std(monkeypatch) -> None:
    calls: dict[str, float] = {}

    def make_blobs(*, n_samples: int, n_features: int, centers: int, cluster_std: float, random_state: int):
        calls["cluster_std"] = cluster_std
        return np.zeros((n_samples, n_features), dtype=np.float32), np.zeros(n_samples, dtype=np.int64)

    sklearn = types.ModuleType("sklearn")
    datasets = types.ModuleType("sklearn.datasets")
    datasets.make_blobs = make_blobs
    sklearn.datasets = datasets
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.datasets", datasets)

    spec = SharedBenchmarkSpec(
        name="blobs_zero_std",
        task="classification",
        source="sklearn",
        dataset="make_blobs",
        input_dim=3,
        num_classes=2,
        centers=2,
        cluster_std=0.0,
        n_samples=8,
    )

    _load_sklearn(spec, seed=42)

    assert calls["cluster_std"] == 0.0
