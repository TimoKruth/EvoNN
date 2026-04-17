"""Shared benchmark loader for compare-side workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_SUPERPROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_CATALOG_ENV_VAR = "COMPARE_CATALOG_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _resolve_catalog_dir() -> Path:
    explicit = os.environ.get(_CATALOG_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()

    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        root = Path(shared_root).expanduser()
    else:
        root = _DEFAULT_SHARED_ROOT
    return root if root.name == "catalog" else root / "catalog"


@dataclass(frozen=True)
class SharedBenchmarkSpec:
    name: str
    task: str
    source: str
    dataset: str | None = None
    source_id: int | None = None
    input_dim: int | None = None
    num_classes: int | None = None
    n_samples: int = 1000
    noise: float = 0.0
    factor: float | None = None
    centers: int | None = None
    cluster_std: float | None = None
    n_informative: int | None = None
    n_redundant: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SharedBenchmarkSpec":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            name=payload["name"],
            task=payload["task"],
            source=payload["source"],
            dataset=payload.get("dataset"),
            source_id=payload.get("source_id"),
            input_dim=payload.get("input_dim"),
            num_classes=payload.get("num_classes"),
            n_samples=payload.get("n_samples", 1000),
            noise=payload.get("noise", 0.0),
            factor=payload.get("factor"),
            centers=payload.get("centers"),
            cluster_std=payload.get("cluster_std"),
            n_informative=payload.get("n_informative"),
            n_redundant=payload.get("n_redundant"),
        )

    def load_data(
        self,
        *,
        seed: int = 42,
        validation_split: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split

        if self.source == "sklearn":
            x, y = _load_sklearn(self, seed=seed)
        elif self.source == "openml":
            x, y = _load_openml(self)
        elif self.source == "image":
            x, y = _load_image(self)
        else:
            raise NotImplementedError(f"Unsupported compare benchmark source: {self.source}")

        stratify = y if self.task == "classification" else None
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=validation_split,
            random_state=seed,
            stratify=stratify,
        )
        y_dtype = np.float32 if self.task == "regression" else np.int64
        return (
            x_train.astype(np.float32, copy=False),
            y_train.astype(y_dtype, copy=False),
            x_val.astype(np.float32, copy=False),
            y_val.astype(y_dtype, copy=False),
        )


def get_benchmark(name: str) -> SharedBenchmarkSpec:
    catalog_dir = _resolve_catalog_dir()
    path = catalog_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark '{name}' not found in shared catalog at {catalog_dir}")
    return SharedBenchmarkSpec.from_yaml(path)


def _load_sklearn(spec: SharedBenchmarkSpec, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import sklearn.datasets as skd

    dataset = spec.dataset
    input_dim = 4 if spec.input_dim is None else spec.input_dim
    num_classes = 2 if spec.num_classes is None else spec.num_classes

    if dataset == "load_iris":
        data = skd.load_iris()
        return data.data, data.target
    if dataset == "load_wine":
        data = skd.load_wine()
        return data.data, data.target
    if dataset == "load_digits":
        data = skd.load_digits()
        return data.data, data.target
    if dataset == "load_breast_cancer":
        data = skd.load_breast_cancer()
        return data.data, data.target
    if dataset == "load_diabetes":
        data = skd.load_diabetes()
        return data.data, data.target
    if dataset == "load_linnerud":
        data = skd.load_linnerud()
        return data.data, data.target[:, 0]
    if dataset == "make_moons":
        return skd.make_moons(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if dataset == "make_circles":
        return skd.make_circles(
            n_samples=spec.n_samples,
            noise=spec.noise,
            factor=0.5 if spec.factor is None else spec.factor,
            random_state=seed,
        )
    if dataset == "make_classification":
        return skd.make_classification(
            n_samples=spec.n_samples,
            n_features=input_dim,
            n_informative=max(2, input_dim // 2) if spec.n_informative is None else spec.n_informative,
            n_redundant=0 if spec.n_redundant is None else spec.n_redundant,
            n_classes=num_classes,
            random_state=seed,
        )
    if dataset == "make_blobs":
        return skd.make_blobs(
            n_samples=spec.n_samples,
            n_features=input_dim,
            centers=num_classes if spec.centers is None else spec.centers,
            cluster_std=1.0 if spec.cluster_std is None else spec.cluster_std,
            random_state=seed,
        )
    if dataset == "make_friedman1":
        return skd.make_friedman1(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if dataset == "make_friedman2":
        return skd.make_friedman2(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if dataset == "make_friedman3":
        return skd.make_friedman3(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if dataset == "make_regression":
        return skd.make_regression(
            n_samples=spec.n_samples,
            n_features=input_dim,
            n_informative=max(2, input_dim // 2) if spec.n_informative is None else spec.n_informative,
            noise=spec.noise,
            random_state=seed,
        )
    raise ValueError(f"Unknown sklearn dataset: {dataset}")


def _load_openml(spec: SharedBenchmarkSpec) -> tuple[np.ndarray, np.ndarray]:
    import openml
    import pandas as pd

    if spec.source_id is None:
        raise ValueError(f"OpenML benchmark '{spec.name}' missing source_id")

    dataset = openml.datasets.get_dataset(spec.source_id, download_data=True)
    x_df, y_series, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    for col in x_df.columns:
        if getattr(x_df[col].dtype, "name", "") == "category" or x_df[col].dtype == object:
            x_df[col] = pd.Categorical(x_df[col]).codes.astype(np.float32)
    x = x_df.to_numpy(dtype=np.float32, na_value=np.nan)
    x = np.nan_to_num(x, nan=0.0)
    y = y_series.to_numpy()
    if spec.task == "classification":
        if y.dtype.kind in {"U", "S", "O"}:
            uniq = {value: index for index, value in enumerate(sorted(set(y)))}
            y = np.asarray([uniq[value] for value in y], dtype=np.int64)
        else:
            y = y.astype(np.int64, copy=False)
    else:
        y = y.astype(np.float32, copy=False)
    return x, y


def _load_image(spec: SharedBenchmarkSpec) -> tuple[np.ndarray, np.ndarray]:
    dataset_name = (spec.dataset or spec.name).lower()
    if dataset_name == "digits" or spec.dataset == "load_digits":
        from sklearn.datasets import load_digits

        data = load_digits()
        return data.data.astype(np.float32), data.target.astype(np.int64)

    import openml

    openml_ids = {"mnist": 554, "fashion_mnist": 40996}
    dataset_id = openml_ids.get(dataset_name)
    if dataset_id is None:
        raise ValueError(f"Unknown image dataset: {dataset_name}")

    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    x_df, y_series, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    x = x_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy()
    if y.dtype.kind in {"U", "S", "O"}:
        y = np.asarray([int(value) for value in y], dtype=np.int64)
    else:
        y = y.astype(np.int64, copy=False)
    return x, y
