"""Run-scoped benchmark data cache for Prism evaluations."""

from __future__ import annotations

import numpy as np

type BenchmarkData = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class BenchmarkDataCache:
    """Reuse loaded benchmark splits across genomes and generations."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], BenchmarkData] = {}

    def resolve(self, spec, *, seed: int = 42) -> BenchmarkData:
        key = (self._spec_key(spec), int(seed))
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
    def _spec_key(spec) -> str:
        benchmark_id = getattr(spec, "id", None) or getattr(spec, "name", None)
        if benchmark_id:
            return str(benchmark_id)
        return repr(spec)

    @staticmethod
    def _load(spec, *, seed: int) -> BenchmarkData:
        if hasattr(spec, "load_data"):
            return spec.load_data(seed=seed)
        return spec.x_train, spec.y_train, spec.x_val, spec.y_val
