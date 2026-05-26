"""Language-modeling benchmark loaders for Prism."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from evonn_shared.benchmarks import shared_benchmarks_root
from evonn_shared.lm_cache import (
    generate_synthetic_lm_dataset as generate_synthetic_lm_dataset,
    load_cached_lm_dataset as _shared_load_cached_lm_dataset,
    resolve_lm_cache_path as _shared_resolve_lm_cache_path,
    split_language_modeling_dataset as split_language_modeling_dataset,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_SUPERPROJECT_ROOT = _PROJECT_ROOT.parent
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_DEFAULT_REPO_CACHE_DIR = _PROJECT_ROOT / "benchmarks" / "lm_cache"
_DEFAULT_LOCAL_CACHE_DIR = Path.home() / ".prism" / "datasets"
_LM_CACHE_ENV_VAR = "PRISM_LM_CACHE_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _search_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get(_SHARED_ROOT_ENV_VAR):
        roots.append(shared_benchmarks_root(env_var=_SHARED_ROOT_ENV_VAR) / "lm_cache")
    roots.extend([_DEFAULT_SHARED_ROOT / "lm_cache", _DEFAULT_REPO_CACHE_DIR, _DEFAULT_LOCAL_CACHE_DIR])
    return roots


def load_cached_lm_dataset(
    dataset: str,
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached LM windows from NPZ and return train/validation splits."""

    return _shared_load_cached_lm_dataset(
        dataset,
        env_var=_LM_CACHE_ENV_VAR,
        search_roots=_search_roots(),
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )


def resolve_lm_cache_path(dataset: str) -> Path:
    """Resolve dataset id or explicit `.npz` path to a cache file."""

    return _shared_resolve_lm_cache_path(dataset, env_var=_LM_CACHE_ENV_VAR, search_roots=_search_roots())
