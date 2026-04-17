"""Language-modeling helpers for Stratograph."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np

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


def _shared_lm_cache_dir() -> Path:
    shared_root = os.environ.get(_SHARED_ROOT_ENV_VAR)
    if shared_root:
        root = Path(shared_root).expanduser()
    else:
        root = _DEFAULT_SHARED_ROOT
    return root / "lm_cache"


def generate_synthetic_lm_dataset(
    *,
    seed: int = 42,
    n_samples: int = 4096,
    context_length: int = 128,
    vocab_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate easy token windows for prototype LM flows."""
    rng = np.random.default_rng(seed=seed)
    sequences = np.zeros((n_samples, context_length + 1), dtype=np.int32)

    for idx in range(n_samples):
        mode = int(rng.integers(0, 4))
        if mode == 0:
            base = rng.integers(0, vocab_size, size=int(rng.integers(3, 8)))
            full = np.tile(base, (context_length + 1) // base.size + 1)[: context_length + 1]
        elif mode == 1:
            start = int(rng.integers(0, vocab_size))
            step = int(rng.integers(1, 5))
            full = np.asarray(
                [(start + step * pos) % vocab_size for pos in range(context_length + 1)],
                dtype=np.int32,
            )
        elif mode == 2:
            a, b = rng.integers(0, vocab_size, size=2)
            full = np.asarray(
                [a if pos % 2 == 0 else b for pos in range(context_length + 1)],
                dtype=np.int32,
            )
        else:
            base = rng.integers(0, vocab_size, size=int(rng.integers(5, 12)))
            full = np.tile(base, (context_length + 1) // base.size + 1)[: context_length + 1]
            noise = rng.random(context_length + 1) < 0.05
            full = np.where(noise, rng.integers(0, vocab_size, size=context_length + 1), full)

        sequences[idx] = full

    return (
        sequences[:, :context_length].astype(np.int32, copy=False),
        sequences[:, 1 : context_length + 1].astype(np.int64, copy=False),
    )


def split_language_modeling_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    validation_split: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split LM windows into train/validation arrays."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("LM feature/target sample counts must match")

    rng = np.random.default_rng(seed=seed)
    order = rng.permutation(x.shape[0])
    val_count = max(1, int(round(x.shape[0] * validation_split)))
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    if train_idx.size == 0:
        raise ValueError("validation_split too large for LM dataset")

    return (
        x[train_idx].astype(np.int32, copy=False),
        y[train_idx].astype(np.int64, copy=False),
        x[val_idx].astype(np.int32, copy=False),
        y[val_idx].astype(np.int64, copy=False),
    )


def load_cached_lm_dataset(
    dataset: str,
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached LM dataset from `.npz`."""
    cache_path = resolve_lm_cache_path(dataset)
    payload = np.load(cache_path)
    x_train = payload["x_train"].astype(np.int32, copy=False)
    y_train = payload["y_train"].astype(np.int64, copy=False)
    x_val = payload["x_val"].astype(np.int32, copy=False)
    y_val = payload["y_val"].astype(np.int64, copy=False)
    if max_train_samples is not None:
        x_train = x_train[:max_train_samples]
        y_train = y_train[:max_train_samples]
    if max_val_samples is not None:
        x_val = x_val[:max_val_samples]
        y_val = y_val[:max_val_samples]
    del max_test_samples
    return x_train, y_train, x_val, y_val


def resolve_lm_cache_path(dataset: str) -> Path:
    """Resolve cache dataset id to concrete file path."""
    candidate = Path(dataset).expanduser()
    if candidate.suffix == ".npz" or candidate.is_absolute() or os.sep in dataset:
        if not candidate.exists():
            raise FileNotFoundError(f"LM cache not found: {candidate}")
        return candidate

    search_roots = _search_roots()
    canonical = _canonical_dataset_name(dataset)
    for root in search_roots:
        path = root / f"{dataset}.npz"
        if path.exists():
            return path
        if canonical != dataset:
            fallback = root / f"{canonical}.npz"
            if fallback.exists():
                return fallback

    roots_text = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"LM cache not found for {dataset}. Checked: {roots_text}. "
        f"Set {_LM_CACHE_ENV_VAR} or place {canonical}.npz in {_shared_lm_cache_dir()}."
    )


def warm_lm_cache(
    datasets: list[str] | None = None,
    *,
    target_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Materialize shared/deprecated LM caches into Stratograph repo cache."""
    target_root = Path(target_dir).expanduser() if target_dir else _DEFAULT_REPO_CACHE_DIR
    target_root.mkdir(parents=True, exist_ok=True)
    datasets = datasets or list(_CANONICAL_DATASETS)
    copied: list[Path] = []

    for dataset in datasets:
        canonical = _canonical_dataset_name(dataset)
        source = _resolve_cache_path_from_roots(canonical, _search_roots(include_repo=False))
        target = target_root / f"{canonical}.npz"
        if target.exists() and not overwrite:
            copied.append(target)
            continue
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def available_lm_caches() -> list[str]:
    """List canonical LM cache names resolvable right now."""
    names: set[str] = set()
    for root in _search_roots():
        if not root.exists():
            continue
        for path in root.glob("*.npz"):
            names.add(path.stem)
    return sorted(names)


def _search_roots(*, include_repo: bool = True) -> list[Path]:
    env_root = os.environ.get(_LM_CACHE_ENV_VAR)
    roots: list[Path] = []
    if env_root:
        root = Path(env_root).expanduser()
        roots.extend([root, root / "datasets"])
    roots.extend([_shared_lm_cache_dir(), _DEFAULT_DEPRECATED_CACHE_DIR])
    if include_repo:
        roots.append(_DEFAULT_REPO_CACHE_DIR)
    roots.append(_DEFAULT_LOCAL_CACHE_DIR)

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def _canonical_dataset_name(dataset: str) -> str:
    if dataset.endswith("_smoke"):
        return dataset.removesuffix("_smoke")
    return dataset


def _resolve_cache_path_from_roots(dataset: str, roots: list[Path]) -> Path:
    for root in roots:
        path = root / f"{dataset}.npz"
        if path.exists():
            return path
    roots_text = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"LM cache not found for {dataset}. Checked: {roots_text}")
