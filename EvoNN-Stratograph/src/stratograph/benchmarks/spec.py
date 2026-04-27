"""Benchmark spec model and minimal loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import train_test_split

from stratograph.benchmarks.lm import (
    generate_synthetic_lm_dataset,
    load_cached_lm_dataset,
    split_language_modeling_dataset,
)


TaskKind = Literal["classification", "regression", "language_modeling"]
MetricDirection = Literal["max", "min"]


class BenchmarkSpec(BaseModel):
    """Benchmark metadata used by runtime and export layers."""

    model_config = ConfigDict(frozen=True)

    name: str
    task: TaskKind
    source: str
    dataset: str | None = None
    url: str | None = None
    path: str | None = None
    target_column: str | None = None
    input_dim: int | None = None
    num_classes: int | None = None
    n_samples: int = 1024
    noise: float = 0.0
    factor: float = 0.5
    centers: int | None = None
    cluster_std: float = 1.0
    n_informative: int | None = None
    n_redundant: int | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    @property
    def metric_name(self) -> str:
        if self.task == "language_modeling":
            return "perplexity"
        if self.task == "regression":
            return "mse"
        return "accuracy"

    @property
    def metric_direction(self) -> MetricDirection:
        return "min" if self.task in {"regression", "language_modeling"} else "max"

    @property
    def model_input_dim(self) -> int:
        return self.input_dim or 8

    @property
    def model_output_dim(self) -> int:
        if self.task == "language_modeling":
            return self.num_classes or 256
        return self.num_classes or 2

    def load_data(
        self,
        *,
        seed: int = 42,
        validation_split: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load small datasets for implemented sources."""
        if self.source == "sklearn":
            x, y = self._load_sklearn(seed=seed)
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
                x_train.astype(np.float32),
                y_train.astype(y_dtype),
                x_val.astype(np.float32),
                y_val.astype(y_dtype),
            )
        if self.source in {"openml", "image"}:
            from stratograph.benchmarks.registry import DatasetRegistry

            return DatasetRegistry().load_data(
                self.dataset or self.name,
                seed=seed,
                validation_split=validation_split,
            )
        if self.source == "lm_synthetic":
            x, y = generate_synthetic_lm_dataset(
                seed=seed,
                n_samples=self.n_samples,
                context_length=self.model_input_dim,
                vocab_size=self.model_output_dim,
            )
            return split_language_modeling_dataset(x, y, seed=seed, validation_split=validation_split)
        if self.source == "lm_cache":
            x_train, y_train, x_val, y_val = load_cached_lm_dataset(
                self.dataset or self.name,
                max_train_samples=self.max_train_samples,
                max_val_samples=self.max_val_samples,
                max_test_samples=self.max_test_samples,
            )
            if self.dataset and self.dataset.endswith("_smoke"):
                x_train = x_train[: min(len(x_train), 1024)]
                y_train = y_train[: min(len(y_train), 1024)]
                x_val = x_val[: min(len(x_val), 256)]
                y_val = y_val[: min(len(y_val), 256)]
            return x_train, y_train, x_val, y_val
        from stratograph.benchmarks.registry import DatasetMeta, DatasetRegistry

        if self.source in {"local", "url"}:
            return DatasetRegistry().load_meta(
                DatasetMeta(
                    name=self.name,
                    source=self.source,
                    url=getattr(self, "url", None),
                    path=getattr(self, "path", None),
                    task=self.task,
                    target_column=getattr(self, "target_column", None),
                    input_dim=self.input_dim,
                    num_classes=self.num_classes,
                    n_samples=self.n_samples,
                ),
                seed=seed,
                validation_split=validation_split,
            )

        return DatasetRegistry().load_data(
            self.dataset or self.name,
            seed=seed,
            validation_split=validation_split,
        )

    def _load_sklearn(self, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        import sklearn.datasets as skd

        if self.dataset == "load_iris":
            data = skd.load_iris()
            return data.data, data.target
        if self.dataset == "load_wine":
            data = skd.load_wine()
            return data.data, data.target
        if self.dataset == "load_digits":
            data = skd.load_digits()
            return data.data, data.target
        if self.dataset == "load_breast_cancer":
            data = skd.load_breast_cancer()
            return data.data, data.target
        if self.dataset == "load_diabetes":
            data = skd.load_diabetes()
            return data.data, data.target
        if self.dataset == "make_moons":
            return skd.make_moons(n_samples=self.n_samples, noise=self.noise, random_state=seed)
        if self.dataset == "make_circles":
            return skd.make_circles(
                n_samples=self.n_samples,
                noise=self.noise,
                factor=self.factor,
                random_state=seed,
            )
        if self.dataset == "make_blobs":
            return skd.make_blobs(
                n_samples=self.n_samples,
                n_features=self.model_input_dim,
                centers=self.centers or self.model_output_dim,
                cluster_std=self.cluster_std,
                random_state=seed,
            )
        if self.dataset == "make_friedman1":
            x, y = skd.make_friedman1(
                n_samples=self.n_samples,
                n_features=self.model_input_dim,
                noise=self.noise,
                random_state=seed,
            )
            return x, y
        raise ValueError(f"Unknown sklearn dataset: {self.dataset}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkSpec":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
