"""Weight inheritance cache with structural and topology hashing."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from topograph.genome.genome import Genome


def structural_hash(genome: Genome) -> str:
    """SHA256 hash of full layer structure (innovation, width, precision, activation, operator)
    plus connection topology. Two genomes with the same structural hash are exact matches
    for weight inheritance.
    """
    parts: list[str] = []

    for layer in sorted(genome.enabled_layers, key=lambda g: g.innovation):
        parts.append(
            f"L:{layer.innovation}:{layer.width}:{layer.weight_bits.value}:"
            f"{layer.activation_bits.value}:{layer.activation.value}:{layer.operator.value}"
        )

    for conn in sorted(genome.enabled_connections, key=lambda g: g.innovation):
        parts.append(f"C:{conn.innovation}:{conn.source}:{conn.target}")

    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def topology_hash(genome: Genome) -> str:
    """SHA256 hash of connectivity only (ignores widths, precision, activation type).
    Used for partial/approximate weight inheritance when an exact match is unavailable.
    """
    parts: list[str] = []

    for layer in sorted(genome.enabled_layers, key=lambda g: g.innovation):
        parts.append(f"L:{layer.innovation}")

    for conn in sorted(genome.enabled_connections, key=lambda g: g.innovation):
        parts.append(f"C:{conn.innovation}:{conn.source}:{conn.target}")

    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _copy_weights(weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in weights.items()}


class WeightCache:
    """FIFO cache for Lamarckian weight inheritance with topology fallback."""

    def __init__(self, max_size: int = 200) -> None:
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._topo_index: dict[str, list[str]] = {}  # topo_hash -> [cache_keys]
        self.max_size = max_size

    def store(
        self, genome: Genome, weights: Mapping[str, np.ndarray], namespace: str = "",
    ) -> None:
        s_hash = structural_hash(genome)
        t_hash = topology_hash(genome)
        key = self._make_key(s_hash, namespace)

        entry = {
            "structural_hash": s_hash,
            "topology_hash": t_hash,
            "weights": _copy_weights(weights),
        }

        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = entry

        # Update topology index
        self._topo_index.setdefault(t_hash, [])
        if key not in self._topo_index[t_hash]:
            self._topo_index[t_hash].append(key)

        # FIFO eviction
        while len(self._cache) > self.max_size:
            evicted_key, evicted_entry = self._cache.popitem(last=False)
            evicted_topo = evicted_entry["topology_hash"]
            if evicted_topo in self._topo_index:
                keys = self._topo_index[evicted_topo]
                if evicted_key in keys:
                    keys.remove(evicted_key)
                if not keys:
                    del self._topo_index[evicted_topo]

    def lookup(
        self, genome: Genome, namespace: str = "",
    ) -> dict[str, np.ndarray] | None:
        """Exact structural match lookup."""
        key = self._make_key(structural_hash(genome), namespace)
        entry = self._cache.get(key)
        if entry is None:
            return None
        self._cache.move_to_end(key)
        return _copy_weights(entry["weights"])

    def lookup_partial(
        self, genome: Genome, namespace: str = "",
    ) -> dict[str, np.ndarray] | None:
        """Topology-only match (ignoring widths/precision). Skips exact matches."""
        s_hash = structural_hash(genome)
        t_hash = topology_hash(genome)
        ns_prefix = f"{namespace}::" if namespace else ""

        candidates = self._topo_index.get(t_hash, [])
        for key in reversed(candidates):
            if ns_prefix and not key.startswith(ns_prefix):
                continue
            entry = self._cache.get(key)
            if entry is None:
                continue
            # Skip exact structural matches (caller should use lookup() for those)
            if entry["structural_hash"] == s_hash:
                continue
            self._cache.move_to_end(key)
            return _copy_weights(entry["weights"])
        return None

    def clear(self) -> None:
        self._cache.clear()
        self._topo_index.clear()

    def __len__(self) -> int:
        return len(self._cache)

    @staticmethod
    def _make_key(s_hash: str, namespace: str) -> str:
        return f"{namespace}::{s_hash}" if namespace else s_hash
