"""Run-scoped benchmark data cache for Prism evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np

type BenchmarkData = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
type _BenchmarkCacheKey = tuple[str, int]

_SPEC_ID_FIELDS = ("id", "name")
_DATA_SPLIT_FIELDS = ("x_train", "y_train", "x_val", "y_val")


class BenchmarkDataCache:
    """Reuse loaded benchmark splits across genomes and generations."""

    def __init__(self) -> None:
        self._cache: dict[_BenchmarkCacheKey, BenchmarkData] = {}

    def resolve(self, spec: Any, *, seed: int = 42) -> BenchmarkData:
        """Return benchmark data for ``spec``, loading it once per seed."""
        key = self._cache_key(spec, seed=seed)
        dataset = self._cache.get(key)
        if dataset is None:
            dataset = self._load(spec, seed=seed)
            self._cache[key] = dataset
        return dataset

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    @staticmethod
    def _cache_key(spec: Any, *, seed: int) -> _BenchmarkCacheKey:
        return BenchmarkDataCache._spec_key(spec), int(seed)

    @staticmethod
    def _spec_key(spec: Any) -> str:
        for field in _SPEC_ID_FIELDS:
            benchmark_id = getattr(spec, field, None)
            if benchmark_id:
                return str(benchmark_id)
        return repr(spec)

    @staticmethod
    def _load(spec: Any, *, seed: int) -> BenchmarkData:
        if hasattr(spec, "load_data"):
            return spec.load_data(seed=seed)
        return tuple(getattr(spec, field) for field in _DATA_SPLIT_FIELDS)
