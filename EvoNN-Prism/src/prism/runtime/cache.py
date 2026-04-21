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
        self._meta: dict[str, dict[str, object]] = {}
        self.max_size = max_size

    def store(self, genome_id: str, model, family: str | None = None) -> None:
        """Snapshot all trainable parameters from an MLX model."""
        import mlx.utils

        snapshot: dict[str, np.ndarray] = {}
        for name, param in mlx.utils.tree_flatten(model.trainable_parameters()):
            snapshot[name] = np.array(param)

        if genome_id in self._cache:
            self._cache.move_to_end(genome_id)
        self._cache[genome_id] = snapshot
        self._meta[genome_id] = {
            "family": family,
            "shapes": {name: tuple(value.shape) for name, value in snapshot.items()},
        }

        # FIFO eviction
        while len(self._cache) > self.max_size:
            old_id, _ = self._cache.popitem(last=False)
            self._meta.pop(old_id, None)

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
        parent_weights = self.lookup(parent_id)
        if parent_weights is None:
            return False

        child_params = _flatten_trainable_parameters(child_model)
        return _apply_matching_weights(parent_weights, child_model, child_params) > 0

    def transfer_best_available(
        self,
        child_model,
        *,
        family: str | None = None,
        preferred_ids: list[str] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> str | None:
        child_params = _flatten_trainable_parameters(child_model)
        exclude_ids = exclude_ids or set()

        candidate_ids: list[str] = []
        seen: set[str] = set()
        for genome_id in preferred_ids or []:
            if genome_id in self._cache and genome_id not in exclude_ids:
                candidate_ids.append(genome_id)
                seen.add(genome_id)
        for genome_id in reversed(self._cache):
            if genome_id in seen or genome_id in exclude_ids:
                continue
            meta = self._meta.get(genome_id, {})
            if family is not None and meta.get("family") not in {None, family}:
                continue
            candidate_ids.append(genome_id)

        best_id = None
        best_score = 0.0
        for genome_id in candidate_ids:
            weights = self._cache.get(genome_id)
            if weights is None:
                continue
            score = _compatibility_score(weights, child_params)
            if score > best_score:
                best_score = score
                best_id = genome_id

        if best_id is None or best_score <= 0.0:
            return None

        parent_weights = self.lookup(best_id)
        if parent_weights is None:
            return None
        transferred = _apply_matching_weights(parent_weights, child_model, child_params)
        return best_id if transferred > 0 else None

    def clear(self) -> None:
        self._cache.clear()
        self._meta.clear()

    def __len__(self) -> int:
        return len(self._cache)


def _flatten_trainable_parameters(model) -> dict[str, np.ndarray]:
    import mlx.utils

    return {
        name: np.array(param)
        for name, param in mlx.utils.tree_flatten(model.trainable_parameters())
    }


def _apply_matching_weights(
    parent_weights: dict[str, np.ndarray],
    child_model,
    child_params: dict[str, np.ndarray],
) -> int:
    import mlx.core as mx

    transferred = 0
    for name, child_param in child_params.items():
        parent_param = parent_weights.get(name)
        if parent_param is None:
            continue
        if parent_param.shape == child_param.shape:
            try:
                child_model.update({name: mx.array(parent_param)})
            except ValueError:
                continue
            transferred += 1
        elif parent_param.ndim == child_param.ndim:
            slices = tuple(slice(0, min(p, c)) for p, c in zip(parent_param.shape, child_param.shape))
            new_param = np.array(child_param)
            overlap = parent_param[slices]
            target_slices = tuple(slice(0, s.stop) for s in slices)
            new_param[target_slices] = overlap
            try:
                child_model.update({name: mx.array(new_param)})
            except ValueError:
                continue
            transferred += 1

    if transferred > 0:
        mx.eval(child_model.parameters())
    return transferred


def _compatibility_score(
    parent_weights: dict[str, np.ndarray],
    child_params: dict[str, np.ndarray],
) -> float:
    score = 0.0
    for name, child_param in child_params.items():
        parent_param = parent_weights.get(name)
        if parent_param is None:
            continue
        if parent_param.shape == child_param.shape:
            score += 2.0
            continue
        if parent_param.ndim != child_param.ndim:
            continue
        overlap = 1.0
        for p, c in zip(parent_param.shape, child_param.shape):
            overlap *= min(p, c) / max(p, c)
        score += max(0.25, overlap)
    return score
