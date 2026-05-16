"""BenchmarkSpec: dataset loading for sklearn, CSV, image, OpenML, and LM sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import DTypeLike
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split


class BenchmarkSpec(BaseModel):
    """Describes a single benchmark dataset and how to load it."""

    id: str = ""
    task: str  # classification | regression | language_modeling
    modality: str = "tabular"  # tabular | image | sequence | text
    input_shape: list[int] = Field(default_factory=list)
    output_dim: int = 1
    metric_name: str = "accuracy"  # accuracy | mse | perplexity
    metric_direction: str = "max"  # max | min
    source: str = "sklearn"  # sklearn | openml | csv | image | lm_synthetic | lm_cache
    dataset: str | None = None
    source_id: int | None = None
    n_samples: int = 1000
    # sklearn-specific params
    noise: float = 0.0
    n_informative: int | None = None
    n_redundant: int | None = None
    n_clusters_per_class: int | None = None
    flip_y: float | None = None
    class_sep: float | None = None
    centers: int | None = None
    cluster_std: float | None = None
    n_targets: int | None = None
    factor: float | None = None
    # aliases from catalog YAML files
    name: str | None = None
    input_dim: int | None = None
    num_classes: int | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    domain: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Reconcile aliases so id, input_shape, output_dim are always set."""
        if self.name and not self.id:
            object.__setattr__(self, "id", self.name)
        if self.input_dim and not self.input_shape:
            object.__setattr__(self, "input_shape", [self.input_dim])
        if self.domain == "text" and self.modality == "tabular":
            object.__setattr__(self, "modality", "text")
        if self.task == "language_modeling":
            object.__setattr__(self, "modality", "text")
        if self.num_classes and self.output_dim <= 1 and self.task in {"classification", "language_modeling"}:
            object.__setattr__(self, "output_dim", self.num_classes)
        # Derive metric from task
        if self.task == "regression" and self.metric_name == "accuracy":
            object.__setattr__(self, "metric_name", "mse")
            object.__setattr__(self, "metric_direction", "min")
        elif self.task == "language_modeling":
            object.__setattr__(self, "metric_name", "perplexity")
            object.__setattr__(self, "metric_direction", "min")

    def load_data(
        self, seed: int = 42, validation_split: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split data. Returns (X_train, y_train, X_val, y_val)."""
        from prism.benchmarks.datasets import load_image, load_openml, load_sklearn
        from prism.benchmarks.lm import (
            generate_synthetic_lm_dataset,
            load_cached_lm_dataset,
            split_language_modeling_dataset,
        )

        if self.source == "sklearn":
            X, y = load_sklearn(self, seed)
        elif self.source == "csv":
            return self._load_csv(seed, validation_split)
        elif self.source == "image":
            X, y = load_image(self)
        elif self.source == "openml":
            X, y = load_openml(self)
        elif self.source == "lm_synthetic":
            X, y = generate_synthetic_lm_dataset(
                seed=seed,
                n_samples=self.n_samples,
                context_length=self.input_dim or 128,
                vocab_size=self.num_classes or self.output_dim or 256,
            )
            return split_language_modeling_dataset(X, y, seed=seed, validation_split=validation_split)
        elif self.source == "lm_cache":
            return load_cached_lm_dataset(
                self.dataset or self.id,
                max_train_samples=self.max_train_samples,
                max_val_samples=self.max_val_samples,
                max_test_samples=self.max_test_samples,
            )
        else:
            raise ValueError(f"Unknown source: {self.source}")

        return _split_cap_and_cast(
            X,
            y,
            seed=seed,
            validation_split=validation_split,
            task=self.task,
            max_train_samples=self.max_train_samples,
            max_val_samples=self.max_val_samples,
        )

    # -- CSV loader ------------------------------------------------------------

    def _load_csv(
        self, seed: int, val_split: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import csv as csv_mod

        with open(self.dataset) as f:
            rows = list(csv_mod.DictReader(f))

        all_cols = list(rows[0].keys())
        n_features = self.input_dim or (len(all_cols) - 1)
        feature_cols = all_cols[:n_features]
        target_col = [c for c in all_cols if c not in feature_cols][0]

        X = np.array([[float(row[c]) for c in feature_cols] for row in rows], dtype=np.float32)

        if self.task == "classification":
            unique = sorted({row[target_col] for row in rows})
            label_map = {t: i for i, t in enumerate(unique)}
            y = np.array([label_map[row[target_col]] for row in rows], dtype=np.int64)
        else:
            y = np.array([float(row[target_col]) for row in rows], dtype=np.float32)

        return _split_cap_and_cast(
            X,
            y,
            seed=seed,
            validation_split=val_split,
            task=self.task,
            max_train_samples=self.max_train_samples,
            max_val_samples=self.max_val_samples,
            x_dtype=np.float32,
        )

    # -- Factory methods -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkSpec:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


def _cap_split(
    x: np.ndarray,
    y: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit <= 0 or len(x) <= limit:
        return x, y
    return x[:limit], y[:limit]


def _split_cap_and_cast(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    validation_split: float,
    task: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    x_dtype: DTypeLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the standard benchmark split, optional caps, and output dtypes."""
    stratify = y if task == "classification" else None
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=validation_split,
        random_state=seed,
        stratify=stratify,
    )
    x_train, y_train = _cap_split(x_train, y_train, max_train_samples)
    x_val, y_val = _cap_split(x_val, y_val, max_val_samples)
    x_dtype = x_dtype or (np.int32 if task == "language_modeling" else np.float32)
    y_dtype = np.float32 if task == "regression" else np.int64
    return (
        x_train.astype(x_dtype),
        y_train.astype(y_dtype),
        x_val.astype(x_dtype),
        y_val.astype(y_dtype),
    )
