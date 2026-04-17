"""Weight inheritance cache for Lamarckian evolution."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


def _copy_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in weights.items()}


class WeightCache:
    """FIFO cache storing trained weight snapshots keyed by genome_id.

    Supports exact-match lookup and shape-compatible transfer from parent
    to child model (Lamarckian weight inheritance).
    """

    def __init__(self, max_size: int = 200) -> None:
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.max_size = max_size

    def store(self, genome_id: str, model) -> None:
        """Snapshot all trainable parameters from an MLX model."""
        import mlx.utils

        snapshot: dict[str, np.ndarray] = {}
        for name, param in mlx.utils.tree_flatten(model.trainable_parameters()):
            snapshot[name] = np.array(param)

        if genome_id in self._cache:
            self._cache.move_to_end(genome_id)
        self._cache[genome_id] = snapshot

        # FIFO eviction
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def lookup(self, genome_id: str) -> dict[str, np.ndarray] | None:
        """Return a copy of cached weights for exact genome_id match."""
        entry = self._cache.get(genome_id)
        if entry is None:
            return None
        self._cache.move_to_end(genome_id)
        return _copy_weights(entry)

    def transfer_weights(self, parent_id: str, child_model) -> bool:
        """Load parent weights into child model where parameter shapes match.

        Returns True if at least one parameter was transferred.
        """
        import mlx.core as mx
        import mlx.utils

        parent_weights = self.lookup(parent_id)
        if parent_weights is None:
            return False

        child_params = dict(mlx.utils.tree_flatten(child_model.trainable_parameters()))
        transferred = 0

        for name, child_param in child_params.items():
            parent_param = parent_weights.get(name)
            if parent_param is None:
                continue
            if parent_param.shape == child_param.shape:
                # Exact shape match: transfer directly
                try:
                    child_model.update({name: mx.array(parent_param)})
                except ValueError:
                    continue
                transferred += 1
            elif parent_param.ndim == child_param.ndim:
                # Partial shape match: transfer overlapping region
                slices = tuple(
                    slice(0, min(p, c))
                    for p, c in zip(parent_param.shape, child_param.shape)
                )
                new_param = np.array(child_param)
                overlap = parent_param[slices]
                target_slices = tuple(
                    slice(0, s.stop) for s in slices
                )
                new_param[target_slices] = overlap
                try:
                    child_model.update({name: mx.array(new_param)})
                except ValueError:
                    continue
                transferred += 1

        if transferred > 0:
            mx.eval(child_model.parameters())

        return transferred > 0

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
