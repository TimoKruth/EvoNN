"""Dataset registry and loaders backed by shared benchmark catalogs."""

from __future__ import annotations

import csv as csv_mod
import os
from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import train_test_split

from stratograph.benchmarks.lm import generate_synthetic_lm_dataset, load_cached_lm_dataset


class DatasetSource(str, Enum):
    SKLEARN = "sklearn"
    OPENML = "openml"
    URL = "url"
    LOCAL = "local"
    IMAGE = "image"
    LM_SYNTHETIC = "lm_synthetic"
    LM_CACHE = "lm_cache"


class DatasetMeta(BaseModel):
    """Catalog metadata for one dataset."""

    model_config = ConfigDict(frozen=True)

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


_SUPERPROJECT_ROOT = Path(__file__).resolve().parents[4]
_LOCAL_CATALOG_DIR = _SUPERPROJECT_ROOT / "EvoNN-Stratograph" / "benchmarks" / "catalog"
_TOPOGRAPH_CATALOG_DIR = _SUPERPROJECT_ROOT / "EvoNN-Topograph" / "benchmarks" / "catalog"
_PRISM_CATALOG_DIR = _SUPERPROJECT_ROOT / "EvoNN-Prism" / "benchmarks" / "catalog"
_CONTENDERS_CATALOG_DIR = _SUPERPROJECT_ROOT / "EvoNN-Contenders" / "benchmarks" / "catalog"
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_CATALOG_ENV_VAR = "STRATOGRAPH_CATALOG_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"
_CACHE_DIR = Path.home() / ".stratograph" / "data"


def _catalog_dirs() -> list[Path]:
    dirs: list[Path] = []
    explicit = os.environ.get(_CATALOG_ENV_VAR)
    if explicit:
        dirs.append(Path(explicit).expanduser())

    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        root = Path(shared_root).expanduser()
    else:
        root = _DEFAULT_SHARED_ROOT
    dirs.append(root if root.name == "catalog" else root / "catalog")

    dirs.extend([
        _LOCAL_CATALOG_DIR,
        _TOPOGRAPH_CATALOG_DIR,
        _PRISM_CATALOG_DIR,
        _CONTENDERS_CATALOG_DIR,
    ])

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


class DatasetRegistry:
    """Resolve benchmark metadata and load arrays from shared boundary sources."""

    def __init__(self) -> None:
        self.catalog: dict[str, DatasetMeta] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        for root in _catalog_dirs():
            if not root.exists():
                continue
            for path in sorted(root.glob("*.yaml")):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not payload or "name" not in payload or "source" not in payload:
                    continue
                meta = DatasetMeta.model_validate(payload)
                self.catalog.setdefault(meta.name, meta)

    def get(self, name: str) -> DatasetMeta:
        try:
            return self.catalog[name]
        except KeyError as exc:
            raise FileNotFoundError(f"Dataset catalog missing: {name}") from exc

    def load_data(
        self,
        name: str,
        *,
        seed: int = 42,
        validation_split: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        meta = self.get(name)
        return self.load_meta(meta, seed=seed, validation_split=validation_split)

    def load_meta(
        self,
        meta: DatasetMeta,
        *,
        seed: int = 42,
        validation_split: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if meta.source == DatasetSource.LM_CACHE:
            return load_cached_lm_dataset(meta.path or meta.name)

        x, y = self._fetch(meta)
        stratify = y if meta.task == "classification" else None
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=validation_split,
            random_state=seed,
            stratify=stratify,
        )
        x_dtype = np.int32 if meta.task == "language_modeling" else np.float32
        y_dtype = np.float32 if meta.task == "regression" else np.int64
        return (
            x_train.astype(x_dtype, copy=False),
            y_train.astype(y_dtype, copy=False),
            x_val.astype(x_dtype, copy=False),
            y_val.astype(y_dtype, copy=False),
        )

    def _fetch(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        if meta.source == DatasetSource.OPENML:
            return self._fetch_openml(meta)
        if meta.source == DatasetSource.IMAGE:
            return self._fetch_image(meta)
        if meta.source == DatasetSource.LM_SYNTHETIC:
            return self._fetch_lm_synthetic(meta)
        if meta.source == DatasetSource.SKLEARN:
            return self._fetch_sklearn(meta)
        if meta.source == DatasetSource.URL:
            return self._fetch_url(meta)
        if meta.source == DatasetSource.LOCAL:
            return self._load_csv_file(Path(meta.path or ""), meta)
        raise ValueError(f"Unsupported source: {meta.source}")

    def _fetch_openml(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        import openml
        import pandas as pd

        dataset = openml.datasets.get_dataset(meta.source_id, download_data=True)
        x_df, y_series, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        for col in x_df.columns:
            if getattr(x_df[col].dtype, "name", "") == "category" or x_df[col].dtype == object:
                x_df[col] = pd.Categorical(x_df[col]).codes.astype(np.float32)
        x = x_df.to_numpy(dtype=np.float32, na_value=np.nan)
        x = np.nan_to_num(x, nan=0.0)
        y = y_series.to_numpy()
        if meta.task == "classification":
            if y.dtype.kind in {"U", "S", "O"}:
                unique = {value: index for index, value in enumerate(sorted(set(y)))}
                y = np.asarray([unique[value] for value in y], dtype=np.int64)
            else:
                y = y.astype(np.int64, copy=False)
        else:
            y = y.astype(np.float32, copy=False)
        return x, y

    def _fetch_image(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        if meta.name == "digits":
            from sklearn.datasets import load_digits

            data = load_digits()
            return data.data.astype(np.float32), data.target.astype(np.int64)

        openml_ids = {"mnist": 554, "fashion_mnist": 40996}
        dataset_id = openml_ids.get(meta.name)
        if dataset_id is None:
            raise ValueError(f"Unknown image dataset: {meta.name}")

        import openml

        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        x_df, y_series, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        x = x_df.to_numpy(dtype=np.float32)
        y = y_series.to_numpy()
        if y.dtype.kind in {"U", "S", "O"}:
            y = np.asarray([int(value) for value in y], dtype=np.int64)
        else:
            y = y.astype(np.int64, copy=False)
        return x, y

    def _fetch_lm_synthetic(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        return generate_synthetic_lm_dataset(
            seed=42,
            n_samples=meta.n_samples or 4096,
            context_length=meta.input_dim or 128,
            vocab_size=meta.num_classes or 256,
        )

    def _fetch_sklearn(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        from stratograph.benchmarks.spec import BenchmarkSpec

        spec_path = next(
            (root / f"{meta.name}.yaml" for root in _catalog_dirs() if (root / f"{meta.name}.yaml").exists()),
            None,
        )
        if spec_path is None:
            raise FileNotFoundError(f"Benchmark catalog YAML missing for {meta.name}")
        spec = BenchmarkSpec.from_yaml(spec_path)
        x_train, y_train, x_val, y_val = spec.load_data(seed=42)
        return np.vstack([x_train, x_val]), np.concatenate([y_train, y_val])

    def _fetch_url(self, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        import urllib.request

        cache_path = _CACHE_DIR / meta.name / "data.csv"
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(meta.url, cache_path)
        return self._load_csv_file(cache_path, meta)

    def _load_csv_file(self, path: Path, meta: DatasetMeta) -> tuple[np.ndarray, np.ndarray]:
        rows = list(csv_mod.DictReader(path.read_text(encoding="utf-8").splitlines()))
        target_col = meta.target_column or list(rows[0].keys())[-1]
        feature_cols = [column for column in rows[0].keys() if column != target_col]
        x = np.asarray(
            [[float(row[column]) for column in feature_cols] for row in rows],
            dtype=np.float32,
        )
        if meta.task == "classification":
            unique = {value: index for index, value in enumerate(sorted({row[target_col] for row in rows}))}
            y = np.asarray([unique[row[target_col]] for row in rows], dtype=np.int64)
        else:
            y = np.asarray([float(row[target_col]) for row in rows], dtype=np.float32)
        return x, y
