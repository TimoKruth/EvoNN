"""Language-modeling benchmark loaders for Prism."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent.parent
_DEFAULT_REPO_CACHE_DIR = _PROJECT_ROOT / "benchmarks" / "lm_cache"
_DEFAULT_LOCAL_CACHE_DIR = Path.home() / ".prism" / "datasets"
_LM_CACHE_ENV_VAR = "PRISM_LM_CACHE_DIR"


def generate_synthetic_lm_dataset(
    *,
    seed: int = 42,
    n_samples: int = 7000,
    context_length: int = 128,
    vocab_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple next-token windows with a few easy patterns."""
    rng = np.random.default_rng(seed=seed)
    sequences = np.zeros((n_samples, context_length + 1), dtype=np.int32)

    for idx in range(n_samples):
        pattern_type = int(rng.integers(0, 4))
        if pattern_type == 0:
            subseq_len = int(rng.integers(3, 8))
            subseq = rng.integers(0, vocab_size, size=subseq_len)
            full = np.tile(subseq, (context_length + 1) // subseq_len + 1)[: context_length + 1]
        elif pattern_type == 1:
            start = int(rng.integers(0, vocab_size))
            step = int(rng.integers(1, 4))
            full = np.array(
                [(start + i * step) % vocab_size for i in range(context_length + 1)],
                dtype=np.int32,
            )
        elif pattern_type == 2:
            a, b = rng.integers(0, vocab_size, size=2)
            full = np.array(
                [a if i % 2 == 0 else b for i in range(context_length + 1)],
                dtype=np.int32,
            )
        else:
            subseq_len = int(rng.integers(4, 12))
            subseq = rng.integers(0, vocab_size, size=subseq_len)
            full = np.tile(subseq, (context_length + 1) // subseq_len + 1)[: context_length + 1]
            noise_mask = rng.random(context_length + 1) < 0.05
            noise_tokens = rng.integers(0, vocab_size, size=context_length + 1)
            full = np.where(noise_mask, noise_tokens, full).astype(np.int32)

        sequences[idx] = full

    x = sequences[:, :context_length].astype(np.int32, copy=False)
    y = sequences[:, 1 : context_length + 1].astype(np.int64, copy=False)
    return x, y


def split_language_modeling_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    validation_split: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle/split LM windows into train/validation sets."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("LM features/targets must have matching sample count")

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(x.shape[0])
    val_count = max(1, int(round(x.shape[0] * validation_split)))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if train_idx.size == 0:
        raise ValueError("validation_split too high for LM dataset")

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
    """Load cached LM windows from NPZ and return train/validation splits."""
    cache_path = resolve_lm_cache_path(dataset)
    payload = np.load(cache_path)

    x_train = payload["x_train"].astype(np.int32, copy=False)
    y_train = payload["y_train"].astype(np.int64, copy=False)
    x_val = payload["x_val"].astype(np.int32, copy=False)
    y_val = payload["y_val"].astype(np.int64, copy=False)

    if max_train_samples is not None and x_train.shape[0] > max_train_samples:
        x_train = x_train[:max_train_samples]
        y_train = y_train[:max_train_samples]
    if max_val_samples is not None and x_val.shape[0] > max_val_samples:
        x_val = x_val[:max_val_samples]
        y_val = y_val[:max_val_samples]
    del max_test_samples

    return x_train, y_train, x_val, y_val


def resolve_lm_cache_path(dataset: str) -> Path:
    """Resolve dataset id or explicit `.npz` path to a cache file."""
    candidate = Path(dataset).expanduser()
    if candidate.suffix == ".npz" or candidate.is_absolute() or os.sep in dataset:
        if not candidate.exists():
            raise FileNotFoundError(f"LM cache not found: {candidate}")
        return candidate

    env_root = os.environ.get(_LM_CACHE_ENV_VAR)
    search_roots: list[Path] = []
    if env_root:
        root = Path(env_root).expanduser()
        search_roots.extend([root, root / "datasets"])
    search_roots.extend([_DEFAULT_REPO_CACHE_DIR, _DEFAULT_LOCAL_CACHE_DIR])

    for root in search_roots:
        path = root / f"{dataset}.npz"
        if path.exists():
            return path

    roots_text = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"LM cache not found for {dataset}. Checked: {roots_text}. "
        f"Set {_LM_CACHE_ENV_VAR} or place {dataset}.npz in {_DEFAULT_REPO_CACHE_DIR}."
    )
