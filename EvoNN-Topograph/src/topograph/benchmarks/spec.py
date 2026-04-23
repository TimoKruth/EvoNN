"""BenchmarkSpec: dataset loading for sklearn, CSV, image, OpenML, and LM sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from topograph.benchmarks.lm import (
    generate_synthetic_lm_dataset,
    load_cached_lm_dataset,
    split_language_modeling_dataset,
)


class BenchmarkSpec(BaseModel):
    name: str
    task: str  # "classification", "regression", or "language_modeling"
    source: str  # "sklearn", "csv", "image", "openml", "lm_synthetic", "lm_cache"
    dataset: str | None = None
    input_dim: int | None = None
    num_classes: int | None = None
    n_samples: int = 1000
    noise: float = 0.0
    # sklearn generator params
    n_informative: int | None = None
    n_redundant: int | None = None
    n_clusters_per_class: int | None = None
    flip_y: float | None = None
    class_sep: float | None = None
    centers: int | None = None
    cluster_std: float | None = None
    n_targets: int | None = None
    factor: float | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    def load_data(
        self, seed: int = 42, validation_split: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split data. Returns (X_train, y_train, X_val, y_val)."""
        if self.source == "sklearn":
            X, y = self._load_sklearn(seed)
        elif self.source == "csv":
            return self._load_csv(seed, validation_split)
        elif self.source == "image":
            from topograph.benchmarks.registry import DatasetRegistry

            registry = DatasetRegistry()
            return registry.load_data(self.dataset or self.name, seed=seed, validation_split=validation_split)
        elif self.source == "openml":
            from topograph.benchmarks.registry import DatasetRegistry

            registry = DatasetRegistry()
            return registry.load_data(self.dataset or self.name, seed=seed, validation_split=validation_split)
        elif self.source == "lm_synthetic":
            x, y = generate_synthetic_lm_dataset(
                seed=seed,
                n_samples=self.n_samples,
                context_length=self.input_dim or 128,
                vocab_size=self.num_classes or 256,
            )
            return split_language_modeling_dataset(
                x, y, seed=seed, validation_split=validation_split,
            )
        elif self.source == "lm_cache":
            return load_cached_lm_dataset(
                self.dataset or self.name,
                max_train_samples=self.max_train_samples,
                max_val_samples=self.max_val_samples,
                max_test_samples=self.max_test_samples,
            )
        else:
            raise ValueError(f"Unknown source: {self.source}")

        stratify = y if self.task == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=seed, stratify=stratify,
        )
        y_dtype = np.float32 if self.task == "regression" else np.int64
        return (
            X_train.astype(np.float32),
            y_train.astype(y_dtype),
            X_val.astype(np.float32),
            y_val.astype(y_dtype),
        )

    # -- sklearn loaders -------------------------------------------------------

    def _load_sklearn(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        import sklearn.datasets as skd

        ds = self.dataset

        # Built-in loaders
        if ds == "load_iris":
            d = skd.load_iris()
            return d.data, d.target
        if ds == "load_wine":
            d = skd.load_wine()
            return d.data, d.target
        if ds == "load_digits":
            d = skd.load_digits()
            return d.data, d.target
        if ds == "load_breast_cancer":
            d = skd.load_breast_cancer()
            return d.data, d.target
        if ds == "load_diabetes":
            d = skd.load_diabetes()
            return d.data, d.target
        if ds == "load_linnerud":
            d = skd.load_linnerud()
            return d.data, d.target[:, 0]

        # Generated classification
        if ds == "make_moons":
            return skd.make_moons(n_samples=self.n_samples, noise=self.noise, random_state=seed)
        if ds == "make_circles":
            return skd.make_circles(
                n_samples=self.n_samples, noise=self.noise,
                factor=self.factor or 0.5, random_state=seed,
            )
        if ds == "make_classification":
            return skd.make_classification(
                n_samples=self.n_samples, n_features=self.input_dim,
                n_informative=self.n_informative or max(2, (self.input_dim or 4) // 2),
                n_redundant=self.n_redundant or 0,
                n_clusters_per_class=self.n_clusters_per_class or 1,
                n_classes=self.num_classes or 2,
                flip_y=self.flip_y or 0.01,
                class_sep=self.class_sep or 1.0,
                random_state=seed,
            )
        if ds == "make_blobs":
            return skd.make_blobs(
                n_samples=self.n_samples, n_features=self.input_dim,
                centers=self.centers or (self.num_classes or 3),
                cluster_std=self.cluster_std or 1.0,
                random_state=seed,
            )
        if ds == "make_gaussian_quantiles":
            return skd.make_gaussian_quantiles(
                n_samples=self.n_samples, n_features=self.input_dim,
                n_classes=self.num_classes or 2, random_state=seed,
            )
        if ds == "make_hastie_10_2":
            X, y = skd.make_hastie_10_2(n_samples=self.n_samples, random_state=seed)
            return X, ((y + 1) / 2).astype(int)

        # Generated regression
        if ds == "make_regression":
            return skd.make_regression(
                n_samples=self.n_samples, n_features=self.input_dim,
                n_informative=self.n_informative or max(2, (self.input_dim or 4) // 2),
                n_targets=self.n_targets or 1,
                noise=self.noise, random_state=seed,
            )
        if ds == "make_friedman1":
            return skd.make_friedman1(n_samples=self.n_samples, noise=self.noise, random_state=seed)
        if ds == "make_friedman2":
            return skd.make_friedman2(n_samples=self.n_samples, noise=self.noise, random_state=seed)
        if ds == "make_friedman3":
            return skd.make_friedman3(n_samples=self.n_samples, noise=self.noise, random_state=seed)
        if ds == "make_sparse_uncorrelated":
            return skd.make_sparse_uncorrelated(
                n_samples=self.n_samples, n_features=self.input_dim, random_state=seed,
            )

        raise ValueError(f"Unknown sklearn dataset: {ds}")

    # -- CSV loader ------------------------------------------------------------

    def _load_csv(
        self, seed: int, val_split: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import csv as csv_mod

        with open(self.dataset) as f:
            rows = list(csv_mod.DictReader(f))

        all_cols = list(rows[0].keys())
        feature_cols = all_cols[: self.input_dim]
        target_col = [c for c in all_cols if c not in feature_cols][0]

        X = np.array([[float(row[c]) for c in feature_cols] for row in rows])

        if self.task == "classification":
            unique = sorted({row[target_col] for row in rows})
            label_map = {t: i for i, t in enumerate(unique)}
            y = np.array([label_map[row[target_col]] for row in rows])
        else:
            y = np.array([float(row[target_col]) for row in rows])

        stratify = y if self.task == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=seed, stratify=stratify,
        )
        y_dtype = np.float32 if self.task == "regression" else np.int64
        return (
            X_train.astype(np.float32),
            y_train.astype(y_dtype),
            X_val.astype(np.float32),
            y_val.astype(y_dtype),
        )

    # -- Factory methods -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkSpec:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        name: str | None = None,
        task: str = "classification",
        target_column: str | None = None,
    ) -> BenchmarkSpec:
        import csv as csv_mod

        path = Path(path)
        with open(path) as f:
            rows = list(csv_mod.DictReader(f))

        if target_column:
            columns = [c for c in rows[0].keys() if c != target_column]
        else:
            all_cols = list(rows[0].keys())
            target_column = all_cols[-1]
            columns = all_cols[:-1]

        input_dim = len(columns)
        num_classes = len({row[target_column] for row in rows}) if task == "classification" else 1

        return cls(
            name=name or path.stem,
            task=task,
            source="csv",
            dataset=str(path),
            input_dim=input_dim,
            num_classes=num_classes,
        )
