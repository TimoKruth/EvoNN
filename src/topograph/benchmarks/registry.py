"""Dataset catalog from YAML files with multi-source loading."""

from __future__ import annotations

import csv as csv_mod
import os
from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from topograph.benchmarks.lm import (
    generate_synthetic_lm_dataset,
    load_cached_lm_dataset,
)


class DatasetSource(str, Enum):
    SKLEARN = "sklearn"
    OPENML = "openml"
    URL = "url"
    LOCAL = "local"
    IMAGE = "image"
    LM_SYNTHETIC = "lm_synthetic"
    LM_CACHE = "lm_cache"


class DatasetMeta(BaseModel):
    name: str
    source: DatasetSource
    source_id: int | None = None
    url: str | None = None
    path: str | None = None
    task: str
    target_column: str | None = None
    input_dim: int | None = None
    num_classes: int | None = None
    n_samples: int | None = None
    description: str = ""
    domain: str = ""
    tags: list[str] = []


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_SUPERPROJECT_ROOT = _PROJECT_ROOT.parent
_LOCAL_CATALOG_DIR = _PROJECT_ROOT / "benchmarks" / "catalog"
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_CATALOG_ENV_VAR = "TOPOGRAPH_CATALOG_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"
CACHE_DIR = Path.home() / ".topograph" / "data"


def _shared_catalog_dir() -> Path:
    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        root = Path(shared_root).expanduser()
    else:
        root = _DEFAULT_SHARED_ROOT
    return root if root.name == "catalog" else root / "catalog"


def _resolve_catalog_dir() -> Path:
    explicit = os.environ.get(_CATALOG_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    shared = _shared_catalog_dir()
    if shared.exists():
        return shared
    return _LOCAL_CATALOG_DIR


class DatasetRegistry:
    """Registry of datasets loaded from YAML catalog files."""

    def __init__(self, catalog_dir: str | None = None) -> None:
        self._catalog_dir = Path(catalog_dir) if catalog_dir else _resolve_catalog_dir()
        self._catalog: dict[str, DatasetMeta] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        if not self._catalog_dir.exists():
            return
        for path in sorted(self._catalog_dir.glob("*.yaml")):
            with open(path) as f:
                data = yaml.safe_load(f)
            if not data or "name" not in data or "source" not in data:
                continue
            try:
                meta = DatasetMeta.model_validate(data)
                self._catalog[meta.name] = meta
            except Exception:
                pass  # skip invalid configs

    def get(self, name: str) -> DatasetMeta:
        if name not in self._catalog:
            raise FileNotFoundError(f"Dataset not found in registry: {name}")
        return self._catalog[name]

    def list(
        self,
        domain: str | None = None,
        task: str | None = None,
        tag: str | None = None,
    ) -> list[DatasetMeta]:
        results = list(self._catalog.values())
        if domain:
            results = [m for m in results if m.domain == domain]
        if task:
            results = [m for m in results if m.task == task]
        if tag:
            results = [m for m in results if tag in m.tags]
        return results

    def load_data(
        self, name: str, seed: int = 42, validation_split: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        meta = self.get(name)
        if meta.source == DatasetSource.LM_CACHE:
            return load_cached_lm_dataset(meta.path or meta.name)

        X, y = self._fetch(meta)

        if meta.task == "classification":
            y = y.astype(np.int64)
            stratify = y
        elif meta.task == "language_modeling":
            y = y.astype(np.int64)
            stratify = None
        else:
            y = y.astype(np.float32)
            stratify = None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=seed, stratify=stratify,
        )
        return (
            X_train.astype(np.int32 if meta.task == "language_modeling" else np.float32),
            y_train,
            X_val.astype(np.int32 if meta.task == "language_modeling" else np.float32),
            y_val,
        )

    def _fetch(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        if meta.source == DatasetSource.OPENML:
            return self._fetch_openml(meta)
        if meta.source == DatasetSource.URL:
            return self._fetch_url(meta)
        if meta.source == DatasetSource.LOCAL:
            return self._load_csv_file(Path(meta.path), meta)
        if meta.source == DatasetSource.SKLEARN:
            return self._fetch_sklearn(meta)
        if meta.source == DatasetSource.IMAGE:
            return self._fetch_image(meta)
        if meta.source == DatasetSource.LM_SYNTHETIC:
            return self._fetch_lm_synthetic(meta)
        raise ValueError(f"Unsupported source: {meta.source}")

    def _fetch_openml(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        import openml
        import pandas as pd

        dataset = openml.datasets.get_dataset(meta.source_id, download_data=True)
        X_df, y_series, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        # Encode categorical features to numeric
        for col in X_df.columns:
            if hasattr(X_df[col], 'cat') or X_df[col].dtype.name == 'category' or X_df[col].dtype == object:
                X_df[col] = pd.Categorical(X_df[col]).codes.astype(np.float32)

        X = X_df.to_numpy(dtype=np.float32, na_value=np.nan)
        # Replace remaining NaN with 0
        X = np.nan_to_num(X, nan=0.0)

        y = y_series.to_numpy()
        if meta.task == "classification":
            if y.dtype.kind in ("U", "S", "O"):
                uniq = sorted(set(y))
                label_map = {v: i for i, v in enumerate(uniq)}
                y = np.array([label_map[v] for v in y], dtype=np.int64)
        else:
            y = y.astype(np.float32)
        return X, y

    def _fetch_image(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        """Load image datasets from sklearn (digits) or fetch via openml (mnist, fashion_mnist)."""
        name = meta.name.lower()
        if name == "digits":
            from sklearn.datasets import load_digits
            data = load_digits()
            return data.data.astype(np.float32), data.target.astype(np.int64)

        # MNIST and Fashion-MNIST via openml
        openml_ids = {"mnist": 554, "fashion_mnist": 40996}
        oid = openml_ids.get(name)
        if oid is None:
            raise ValueError(f"Unknown image dataset: {name}")

        import openml
        dataset = openml.datasets.get_dataset(oid, download_data=True)
        X_df, y_series, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X_df.to_numpy(dtype=np.float32)
        y = y_series.to_numpy()
        if y.dtype.kind in ("U", "S", "O"):
            y = np.array([int(v) for v in y], dtype=np.int64)
        return X, y

    def _fetch_url(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        import urllib.request

        cache_path = CACHE_DIR / meta.name / "data.csv"
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(meta.url, cache_path)
        return self._load_csv_file(cache_path, meta)

    def _fetch_lm_synthetic(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        return generate_synthetic_lm_dataset(
            seed=42,
            n_samples=meta.n_samples or 7000,
            context_length=meta.input_dim or 128,
            vocab_size=meta.num_classes or 256,
        )

    def _fetch_sklearn(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        from topograph.benchmarks.spec import BenchmarkSpec

        spec_path = self._catalog_dir / f"{meta.name}.yaml"
        if spec_path.exists():
            spec = BenchmarkSpec.from_yaml(spec_path)
            X_train, y_train, X_val, y_val = spec.load_data(seed=42)
            return np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
        raise FileNotFoundError(f"No catalog YAML for sklearn dataset: {meta.name}")

    def _load_csv_file(
        self, path: Path, meta: DatasetMeta,
    ) -> tuple[np.ndarray, np.ndarray]:
        with open(path) as f:
            rows = list(csv_mod.DictReader(f))

        target_col = meta.target_column or list(rows[0].keys())[-1]
        feature_cols = [c for c in rows[0].keys() if c != target_col]

        X = np.array(
            [[float(row[c]) for c in feature_cols] for row in rows], dtype=np.float32,
        )
        if meta.task == "classification":
            unique = sorted({row[target_col] for row in rows})
            label_map = {t: i for i, t in enumerate(unique)}
            y = np.array([label_map[row[target_col]] for row in rows], dtype=np.int64)
        else:
            y = np.array([float(row[target_col]) for row in rows], dtype=np.float32)

        return X, y
