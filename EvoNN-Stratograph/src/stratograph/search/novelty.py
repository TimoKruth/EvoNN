"""Novelty descriptors and archive helpers."""

from __future__ import annotations

import math

from stratograph.genome.models import HierarchicalGenome


def descriptor(genome: HierarchicalGenome) -> tuple[float, float, float, float]:
    """Compact behavior descriptor for novelty/QD bookkeeping."""
    return (
        float(genome.macro_depth),
        float(genome.average_cell_depth),
        float(len(genome.cell_library)),
        float(genome.reuse_ratio),
    )


def novelty_score(
    current: tuple[float, float, float, float],
    archive: list[tuple[float, float, float, float]],
    *,
    k: int = 4,
) -> float:
    """Average distance to nearest k descriptors."""
    if not archive:
        return 0.0
    distances = sorted(_distance(current, other) for other in archive)
    limit = min(k, len(distances))
    return sum(distances[:limit]) / limit


def niche_key(current: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Bucket descriptor into a coarse niche."""
    return (
        int(current[0]),
        int(current[1]),
        int(current[2]),
        int(current[3] * 10),
    )


def _distance(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
