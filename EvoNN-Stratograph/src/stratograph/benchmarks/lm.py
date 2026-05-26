"""Language-modeling helpers for Stratograph."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from evonn_shared.benchmarks import shared_benchmarks_root
from evonn_shared.lm_cache import (
    available_lm_caches as _shared_available_lm_caches,
    generate_synthetic_lm_dataset as generate_synthetic_lm_dataset,
    load_cached_lm_dataset as _shared_load_cached_lm_dataset,
    resolve_lm_cache_path as _shared_resolve_lm_cache_path,
    split_language_modeling_dataset as split_language_modeling_dataset,
    warm_lm_cache as _shared_warm_lm_cache,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_SUPERPROJECT_ROOT = _PROJECT_ROOT.parent
_DEFAULT_SHARED_ROOT = _SUPERPROJECT_ROOT / "shared-benchmarks"
_DEFAULT_REPO_CACHE_DIR = _PROJECT_ROOT / "benchmarks" / "lm_cache"
_DEFAULT_LOCAL_CACHE_DIR = Path.home() / ".stratograph" / "datasets"
_DEFAULT_DEPRECATED_CACHE_DIR = _SUPERPROJECT_ROOT / "deprecated" / "EvoNN" / ".cache" / "evonn" / "datasets"
_LM_CACHE_ENV_VAR = "STRATOGRAPH_LM_CACHE_DIR"
_SHARED_ROOT_ENV_VAR = "EVONN_SHARED_BENCHMARKS_DIR"
_CANONICAL_DATASETS = ("tinystories_lm", "wikitext2_lm")


def _search_roots(*, include_repo: bool = True) -> list[Path]:
    roots: list[Path] = []
    if os.environ.get(_SHARED_ROOT_ENV_VAR):
        roots.append(shared_benchmarks_root(env_var=_SHARED_ROOT_ENV_VAR) / "lm_cache")
    roots.extend([_DEFAULT_SHARED_ROOT / "lm_cache", _DEFAULT_DEPRECATED_CACHE_DIR])
    if include_repo:
        roots.append(_DEFAULT_REPO_CACHE_DIR)
    roots.append(_DEFAULT_LOCAL_CACHE_DIR)
    return roots


def load_cached_lm_dataset(
    dataset: str,
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached LM dataset from `.npz`."""

    return _shared_load_cached_lm_dataset(
        dataset,
        env_var=_LM_CACHE_ENV_VAR,
        search_roots=_search_roots(),
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )


def resolve_lm_cache_path(dataset: str) -> Path:
    """Resolve cache dataset id to concrete file path."""

    return _shared_resolve_lm_cache_path(dataset, env_var=_LM_CACHE_ENV_VAR, search_roots=_search_roots())


def warm_lm_cache(
    datasets: list[str] | None = None,
    *,
    target_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Materialize shared/deprecated LM caches into Stratograph repo cache."""

    return _shared_warm_lm_cache(
        datasets or list(_CANONICAL_DATASETS),
        target_dir=target_dir or _DEFAULT_REPO_CACHE_DIR,
        env_var=_LM_CACHE_ENV_VAR,
        search_roots=_search_roots(include_repo=False),
        overwrite=overwrite,
    )


def available_lm_caches() -> list[str]:
    """List canonical LM cache names resolvable right now."""

    return _shared_available_lm_caches(env_var=_LM_CACHE_ENV_VAR, search_roots=_search_roots())
