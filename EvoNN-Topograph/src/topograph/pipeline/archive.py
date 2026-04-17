"""Unified archive module: novelty, MAP-Elites, benchmark elites, and behavior."""

from __future__ import annotations

import copy
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from topograph.genome import Genome
from topograph.genome.codec import dict_to_genome, genome_to_dict
from topograph.genome.genome import INPUT_INNOVATION, OUTPUT_INNOVATION


# ===========================================================================
# Topology helpers for compute_behavior
# ===========================================================================


def _build_adjacency(genome: Genome) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for c in genome.enabled_connections:
        adj[c.source].append(c.target)
    return dict(adj)


def _all_nodes(adj: dict[int, list[int]]) -> set[int]:
    nodes: set[int] = set()
    for src, targets in adj.items():
        nodes.add(src)
        nodes.update(targets)
    return nodes


def _longest_path_depths(adj: dict[int, list[int]]) -> dict[int, int]:
    nodes = _all_nodes(adj)
    if INPUT_INNOVATION not in nodes:
        return {}

    in_degree: dict[int, int] = defaultdict(int)
    for node in nodes:
        in_degree.setdefault(node, 0)
    for targets in adj.values():
        for tgt in targets:
            in_degree[tgt] += 1

    dist: dict[int, int] = {node: -1 for node in nodes}
    dist[INPUT_INNOVATION] = 0

    queue: deque[int] = deque(n for n in nodes if in_degree[n] == 0)
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, []):
            if dist[node] >= 0:
                dist[neighbor] = max(dist[neighbor], dist[node] + 1)
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return {n: d for n, d in dist.items() if d >= 0}


def _dag_depth(genome: Genome) -> int:
    return _longest_path_depths(_build_adjacency(genome)).get(OUTPUT_INNOVATION, 0)


def _dag_width_profile(genome: Genome) -> list[int]:
    dist = _longest_path_depths(_build_adjacency(genome))
    max_d = max(dist.values(), default=-1)
    if max_d < 0:
        return []
    profile = [0] * (max_d + 1)
    for d in dist.values():
        profile[d] += 1
    return profile


def _skip_connection_count(genome: Genome) -> int:
    depth_map = _longest_path_depths(_build_adjacency(genome))
    count = 0
    for c in genome.enabled_connections:
        sd = depth_map.get(c.source)
        td = depth_map.get(c.target)
        if sd is not None and td is not None and td - sd > 1:
            count += 1
    return count


def _is_reachable(
    adj: dict[int, list[int]], source: int, target: int, valid: set[int]
) -> bool:
    if source not in valid or target not in valid:
        return False
    visited: set[int] = set()
    q: deque[int] = deque([source])
    while q:
        node = q.popleft()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        for nb in adj.get(node, []):
            if nb in valid and nb not in visited:
                q.append(nb)
    return False


def _bottleneck_count(genome: Genome) -> int:
    adj = _build_adjacency(genome)
    nodes = _all_nodes(adj)
    if INPUT_INNOVATION not in nodes or OUTPUT_INNOVATION not in nodes:
        return 0
    if not _is_reachable(adj, INPUT_INNOVATION, OUTPUT_INNOVATION, nodes):
        return 0

    layer_inns = {g.innovation for g in genome.enabled_layers}
    count = 0
    for inn in layer_inns:
        filtered: dict[int, list[int]] = {}
        for src, targets in adj.items():
            if src == inn:
                continue
            f = [t for t in targets if t != inn]
            if f:
                filtered[src] = f
        remaining = nodes - {inn}
        if not _is_reachable(filtered, INPUT_INNOVATION, OUTPUT_INNOVATION, remaining):
            count += 1
    return count


# ===========================================================================
# Behavior vector
# ===========================================================================


def compute_behavior(genome: Genome) -> np.ndarray:
    """Project genome into an 8-D float32 behavior descriptor.

    Dimensions: depth, max_width, skip_connections, bottleneck_count,
    total_layers, total_connections, connectivity_ratio, has_experts.
    """
    depth = _dag_depth(genome)
    wp = _dag_width_profile(genome)
    max_width = max(wp) if wp else 0
    skips = _skip_connection_count(genome)
    bottlenecks = _bottleneck_count(genome)
    n_layers = len(genome.enabled_layers)
    n_conns = len(genome.enabled_connections)
    max_possible = n_layers * (n_layers + 1) / 2 if n_layers > 0 else 1
    connectivity = n_conns / max_possible
    has_experts = 1.0 if len(genome.experts) > 0 else 0.0

    return np.array(
        [depth, max_width, skips, bottlenecks, n_layers, n_conns, connectivity, has_experts],
        dtype=np.float32,
    )


# ===========================================================================
# Novelty archive
# ===========================================================================


class NoveltyArchive:
    """Fixed-size FIFO archive with KNN novelty scoring."""

    def __init__(self, max_size: int = 5000, k: int = 15) -> None:
        self.max_size = max_size
        self.k = k
        self._behaviors: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._behaviors)

    @property
    def behaviors(self) -> list[np.ndarray]:
        return self._behaviors

    def add(self, behavior: np.ndarray) -> None:
        self._behaviors.append(np.array(behavior, dtype=np.float32, copy=True))
        if len(self._behaviors) > self.max_size:
            self._behaviors.pop(0)

    def compute_novelty(
        self,
        behavior: np.ndarray,
        population_behaviors: list[np.ndarray] | None = None,
    ) -> float:
        """KNN distance from behavior to archive + population, normalized."""
        pool = list(self._behaviors)
        if population_behaviors:
            pool.extend(np.array(b, dtype=np.float32, copy=False) for b in population_behaviors)
        if not pool:
            return 0.0

        ref = np.vstack(pool)
        scale = np.std(ref, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)

        query = np.array(behavior, dtype=np.float32, copy=False)
        distances: list[float] = []
        for candidate in pool:
            diff = (query - candidate) / scale
            d = float(np.linalg.norm(diff))
            if d > 1e-12:
                distances.append(d)

        if not distances:
            return 0.0
        distances.sort()
        k = min(self.k, len(distances))
        return float(np.mean(distances[:k]))

    def to_dict(self) -> dict[str, object]:
        return {
            "max_size": self.max_size,
            "k": self.k,
            "behaviors": [b.tolist() for b in self._behaviors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "NoveltyArchive":
        archive = cls(
            max_size=int((data or {}).get("max_size", 5000)),
            k=int((data or {}).get("k", 15)),
        )
        for behavior in (data or {}).get("behaviors", []):
            archive.add(np.array(behavior, dtype=np.float32))
        return archive


# ===========================================================================
# MAP-Elites archive
# ===========================================================================


def _bucket(value: float, thresholds: list[float]) -> int:
    for idx, t in enumerate(thresholds):
        if value <= t:
            return idx
    return len(thresholds)


@dataclass(frozen=True)
class ArchiveElite:
    genome: Genome
    behavior: np.ndarray
    fitness: float
    niche: tuple[int, int, int, int]


class MAPElitesArchive:
    """6x6x6x6 discretized grid archive (1296 niches)."""

    def __init__(self) -> None:
        self._elites: dict[tuple[int, int, int, int], ArchiveElite] = {}

    def __len__(self) -> int:
        return len(self._elites)

    @staticmethod
    def behavior_to_niche(behavior: np.ndarray) -> tuple[int, int, int, int]:
        """Map 8-D behavior to a 4-D niche index.

        Depth buckets: [0-2, 2-4, 4-6, 6-8, 8-12, 12+]
        Width buckets: [0-1, 1-2, 2-3, 3-4, 4-6, 6+]  (log-scaled width / 64)
        Skip conns:    [0, 1, 2, 4, 8, 8+]
        Connectivity:  [0-0.15, 0.15-0.3, 0.3-0.45, 0.45-0.6, 0.6-0.8, 0.8+]
        """
        depth = float(behavior[0])
        max_width = float(behavior[1])
        skips = float(behavior[2])
        connectivity = float(behavior[6])
        return (
            _bucket(depth, [2, 4, 6, 8, 12]),
            _bucket(max_width, [1, 2, 3, 4, 6]),
            _bucket(skips, [0, 1, 2, 4, 8]),
            _bucket(connectivity, [0.15, 0.3, 0.45, 0.6, 0.8]),
        )

    def add(self, genome: Genome, behavior: np.ndarray, fitness: float) -> bool:
        """Insert or replace elite for a niche. Lower fitness is better.

        Returns True if the elite was inserted or replaced.
        """
        niche = self.behavior_to_niche(behavior)
        current = self._elites.get(niche)
        if current is not None and current.fitness <= fitness:
            return False
        self._elites[niche] = ArchiveElite(
            genome=copy.deepcopy(genome),
            behavior=np.array(behavior, dtype=np.float32, copy=True),
            fitness=float(fitness),
            niche=niche,
        )
        return True

    def sample(self, count: int, rng: random.Random) -> list[Genome]:
        """Sample elites uniformly across occupied niches."""
        if count <= 0 or not self._elites:
            return []
        elites = list(self._elites.values())
        if count >= len(elites):
            return [copy.deepcopy(e.genome) for e in elites]
        return [copy.deepcopy(e.genome) for e in rng.sample(elites, count)]

    def entries(self) -> list[ArchiveElite]:
        """Return archive entries in stable niche order."""
        return [self._elites[k] for k in sorted(self._elites)]

    def clear(self) -> None:
        self._elites.clear()

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [
                {
                    "genome": genome_to_dict(entry.genome),
                    "behavior": entry.behavior.tolist(),
                    "fitness": entry.fitness,
                    "niche": list(entry.niche),
                }
                for entry in self.entries()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "MAPElitesArchive":
        archive = cls()
        for raw in (data or {}).get("entries", []):
            if not isinstance(raw, dict):
                continue
            genome = dict_to_genome(raw["genome"])
            behavior = np.array(raw["behavior"], dtype=np.float32)
            archive.add(genome, behavior, float(raw["fitness"]))
        return archive


# ===========================================================================
# Benchmark elite archive
# ===========================================================================


@dataclass
class BenchmarkElite:
    benchmark_name: str
    genome_idx: int
    fitness: float
    generation: int


@dataclass
class BenchmarkEliteArchive:
    """Tracks the best genome per benchmark across generations."""

    elites: dict[str, BenchmarkElite] = field(default_factory=dict)

    def update(
        self, benchmark_name: str, genome_idx: int, fitness: float, generation: int
    ) -> bool:
        """Update the elite for a benchmark if this genome is better. Returns True if updated."""
        current = self.elites.get(benchmark_name)
        if current is None or fitness < current.fitness:
            self.elites[benchmark_name] = BenchmarkElite(
                benchmark_name=benchmark_name,
                genome_idx=genome_idx,
                fitness=fitness,
                generation=generation,
            )
            return True
        return False

    def get_elite_indices(self) -> set[int]:
        """Return the set of genome indices that are elites for at least one benchmark."""
        return {e.genome_idx for e in self.elites.values()}

    def to_dict(self) -> dict[str, object]:
        return {
            "elites": {
                name: {
                    "benchmark_name": elite.benchmark_name,
                    "genome_idx": elite.genome_idx,
                    "fitness": elite.fitness,
                    "generation": elite.generation,
                }
                for name, elite in self.elites.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "BenchmarkEliteArchive":
        archive = cls()
        for name, raw in ((data or {}).get("elites", {}) or {}).items():
            if not isinstance(raw, dict):
                continue
            archive.elites[name] = BenchmarkElite(
                benchmark_name=str(raw.get("benchmark_name", name)),
                genome_idx=int(raw.get("genome_idx", 0)),
                fitness=float(raw.get("fitness", float("inf"))),
                generation=int(raw.get("generation", 0)),
            )
        return archive
