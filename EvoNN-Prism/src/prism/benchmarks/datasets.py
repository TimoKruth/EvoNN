"""Data loading from sklearn, OpenML, image sources, and YAML catalog.

Consolidates all dataset loading into one module with clean functions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import yaml

from prism.benchmarks.spec import BenchmarkSpec

logger = logging.getLogger(__name__)

# Default catalog dir: <project_root>/benchmarks/catalog/
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_SUPERPROJECT_ROOT = _PROJECT_ROOT.parent
_LOCAL_CATALOG_DIR = _PROJECT_ROOT / "benchmarks" / "catalog"
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_DEFAULT_CATALOG_DIR = _LOCAL_CATALOG_DIR
_CATALOG_ENV_VAR = "PRISM_CATALOG_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _shared_catalog_dir() -> Path:
    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        root = Path(shared_root).expanduser()
    else:
        root = _DEFAULT_SHARED_ROOT
    return root if root.name == "catalog" else root / "catalog"


def _resolve_catalog_dir() -> Path:
    """Return configured catalog directory with shared-root-first fallback."""
    explicit = os.environ.get(_CATALOG_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()

    shared = _shared_catalog_dir()
    if shared.exists():
        return shared
    return _LOCAL_CATALOG_DIR


def _catalog_missing_message(cat_dir: Path) -> str:
    return (
        f"Benchmark catalog not found: {cat_dir}. "
        f"Add YAML specs under {_DEFAULT_CATALOG_DIR} or set {_CATALOG_ENV_VAR}."
    )


def _benchmark_missing_message(name: str, cat_dir: Path) -> str:
    return (
        f"Benchmark '{name}' not found in catalog at {cat_dir}. "
        f"Add {name}.yaml there or set {_CATALOG_ENV_VAR} to another catalog directory."
    )


# ---------------------------------------------------------------------------
# sklearn loaders
# ---------------------------------------------------------------------------


def load_sklearn(spec: BenchmarkSpec, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset from sklearn generators or built-in loaders."""
    import sklearn.datasets as skd

    ds = spec.dataset

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

    input_dim = spec.input_dim or (spec.input_shape[0] if spec.input_shape else 4)
    num_classes = spec.num_classes or spec.output_dim or 2

    # Generated classification
    if ds == "make_moons":
        return skd.make_moons(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if ds == "make_circles":
        return skd.make_circles(
            n_samples=spec.n_samples, noise=spec.noise,
            factor=spec.factor or 0.5, random_state=seed,
        )
    if ds == "make_classification":
        return skd.make_classification(
            n_samples=spec.n_samples, n_features=input_dim,
            n_informative=spec.n_informative or max(2, input_dim // 2),
            n_redundant=spec.n_redundant or 0,
            n_clusters_per_class=spec.n_clusters_per_class or 1,
            n_classes=num_classes,
            flip_y=spec.flip_y or 0.01,
            class_sep=spec.class_sep or 1.0,
            random_state=seed,
        )
    if ds == "make_blobs":
        return skd.make_blobs(
            n_samples=spec.n_samples, n_features=input_dim,
            centers=spec.centers or num_classes,
            cluster_std=spec.cluster_std or 1.0,
            random_state=seed,
        )
    if ds == "make_gaussian_quantiles":
        return skd.make_gaussian_quantiles(
            n_samples=spec.n_samples, n_features=input_dim,
            n_classes=num_classes, random_state=seed,
        )
    if ds == "make_hastie_10_2":
        X, y = skd.make_hastie_10_2(n_samples=spec.n_samples, random_state=seed)
        return X, ((y + 1) / 2).astype(int)

    # Generated regression
    if ds == "make_regression":
        return skd.make_regression(
            n_samples=spec.n_samples, n_features=input_dim,
            n_informative=spec.n_informative or max(2, input_dim // 2),
            n_targets=spec.n_targets or 1,
            noise=spec.noise, random_state=seed,
        )
    if ds == "make_friedman1":
        return skd.make_friedman1(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if ds == "make_friedman2":
        return skd.make_friedman2(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if ds == "make_friedman3":
        return skd.make_friedman3(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    if ds == "make_sparse_uncorrelated":
        return skd.make_sparse_uncorrelated(
            n_samples=spec.n_samples, n_features=input_dim, random_state=seed,
        )

    raise ValueError(f"Unknown sklearn dataset: {ds}")


# ---------------------------------------------------------------------------
# OpenML loader
# ---------------------------------------------------------------------------


def load_openml(spec: BenchmarkSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load an OpenML dataset by source_id, with categorical encoding."""
    source_id = spec.source_id
    if source_id is None:
        raise ValueError(f"OpenML benchmark '{spec.id}' missing source_id")

    try:
        import openml
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "OpenML support requires: uv sync --extra data"
        ) from exc

    logger.info("Downloading OpenML dataset %d for %s", source_id, spec.id)
    dataset = openml.datasets.get_dataset(source_id, download_data=True)
    X_df, y_series, categorical_indicator, _ = dataset.get_data(
        target=dataset.default_target_attribute,
    )

    # Encode categorical features to numeric
    for col in X_df.columns:
        if (
            hasattr(X_df[col], "cat")
            or X_df[col].dtype.name == "category"
            or X_df[col].dtype == object
        ):
            X_df[col] = pd.Categorical(X_df[col]).codes.astype(np.float32)

    X = X_df.to_numpy(dtype=np.float32, na_value=np.nan)
    # Replace remaining NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    y = y_series.to_numpy()
    if spec.task == "classification":
        if y.dtype.kind in ("U", "S", "O"):
            uniq = sorted(set(y))
            label_map = {v: i for i, v in enumerate(uniq)}
            y = np.array([label_map[v] for v in y], dtype=np.int64)
        else:
            y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------


def load_image(spec: BenchmarkSpec) -> tuple[np.ndarray, np.ndarray]:
    """Load image datasets: digits via sklearn, mnist/fashion_mnist via openml."""
    name = (spec.dataset or spec.id).lower()

    if name == "digits" or spec.dataset == "load_digits":
        from sklearn.datasets import load_digits
        data = load_digits()
        return data.data.astype(np.float32), data.target.astype(np.int64)

    # MNIST and Fashion-MNIST via OpenML
    openml_ids = {"mnist": 554, "fashion_mnist": 40996}
    oid = openml_ids.get(name)
    if oid is None:
        raise ValueError(f"Unknown image dataset: {name}")

    try:
        import openml
    except ImportError as exc:
        raise ImportError(
            "Image datasets (MNIST, Fashion-MNIST) require: uv sync --extra data"
        ) from exc

    dataset = openml.datasets.get_dataset(oid, download_data=True)
    X_df, y_series, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = X_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy()
    if y.dtype.kind in ("U", "S", "O"):
        y = np.array([int(v) for v in y], dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_catalog(name: str, catalog_dir: Path | None = None) -> BenchmarkSpec:
    """Load a single BenchmarkSpec from a catalog YAML file by name."""
    cat_dir = catalog_dir or _resolve_catalog_dir()
    if not cat_dir.exists():
        raise FileNotFoundError(_catalog_missing_message(cat_dir))
    path = cat_dir / f"{name}.yaml"
    if path.exists():
        return BenchmarkSpec.from_yaml(path)
    raise FileNotFoundError(_benchmark_missing_message(name, cat_dir))


def get_benchmark(name: str, catalog_dir: Path | None = None) -> BenchmarkSpec:
    """Get a BenchmarkSpec by name. Tries catalog YAML first.

    Also checks reverse canonical ID mapping as a fallback.
    """
    from prism.benchmarks.parity import _REVERSE_IDS

    cat_dir = catalog_dir or _resolve_catalog_dir()
    if not cat_dir.exists():
        raise FileNotFoundError(_catalog_missing_message(cat_dir))

    # Direct lookup
    path = cat_dir / f"{name}.yaml"
    if path.exists():
        return BenchmarkSpec.from_yaml(path)

    # Reverse canonical lookup
    native = _REVERSE_IDS.get(name)
    if native:
        alt_path = cat_dir / f"{native}.yaml"
        if alt_path.exists():
            return BenchmarkSpec.from_yaml(alt_path)

    raise FileNotFoundError(_benchmark_missing_message(name, cat_dir))


def list_benchmarks(catalog_dir: Path | None = None) -> list[str]:
    """List all benchmark names available in the catalog."""
    cat_dir = catalog_dir or _resolve_catalog_dir()
    if not cat_dir.exists():
        return []

    names: list[str] = []
    for p in sorted(cat_dir.glob("*.yaml")):
        with open(p) as f:
            data = yaml.safe_load(f)
        if data and "name" in data:
            names.append(p.stem)
    return names
