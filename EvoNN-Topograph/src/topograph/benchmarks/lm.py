"""Language modeling benchmark loaders for Topograph."""

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

SUPERPROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SHARED_CACHE_DIR = SUPERPROJECT_ROOT / "EvoNN" / ".cache" / "evonn" / "datasets"
DEFAULT_REPO_SHARED_CACHE_DIR = SUPERPROJECT_ROOT / "shared-benchmarks" / "lm_cache"
DEFAULT_LOCAL_CACHE_DIR = Path.home() / ".topograph" / "datasets"
_LM_CACHE_ENV_VAR = "TOPOGRAPH_LM_CACHE_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"


def _search_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get(_SHARED_ROOT_ENV_VAR):
        roots.append(shared_benchmarks_root(env_var=_SHARED_ROOT_ENV_VAR) / "lm_cache")
    roots.extend([DEFAULT_SHARED_CACHE_DIR, DEFAULT_REPO_SHARED_CACHE_DIR, DEFAULT_LOCAL_CACHE_DIR])
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
    """Resolve dataset ID or explicit `.npz` path to a cache file."""

    return _shared_resolve_lm_cache_path(dataset, env_var=_LM_CACHE_ENV_VAR, search_roots=_search_roots())
