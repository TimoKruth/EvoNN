"""Selection operators for evolutionary search."""

from __future__ import annotations

import math
import random

from topograph.genome import Genome


def rank_based_select(
    population: list[Genome],
    fitnesses: list[float],
    count: int,
    rng: random.Random,
) -> list[Genome]:
    """Rank-based roulette wheel selection (lower fitness = better).

    Best genome gets weight n, second gets n-1, ..., worst gets 1.
    """
    paired = sorted(zip(fitnesses, population), key=lambda x: x[0])
    n = len(paired)
    total = n * (n + 1) / 2
    weights = [(n - i) / total for i in range(n)]

    selected: list[Genome] = []
    for _ in range(count):
        r = rng.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                selected.append(paired[i][1])
                break
        else:
            selected.append(paired[-1][1])
    return selected


def non_dominated_sort(
    fitnesses: list[float],
    model_bytes: list[int],
) -> list[list[int]]:
    """NSGA-II Pareto front decomposition over two objectives (both minimized).

    Returns list of fronts, each front is a list of population indices.
    """
    n = len(fitnesses)

    def dominates(i: int, j: int) -> bool:
        fi, fj = fitnesses[i], fitnesses[j]
        bi, bj = model_bytes[i], model_bytes[j]
        not_worse = fi <= fj and bi <= bj
        better_one = fi < fj or bi < bj
        return not_worse and better_one

    domination_count = [0] * n
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(i, j):
                dominated_by[i].append(j)
            elif dominates(j, i):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return [f for f in fronts if f]


def crowding_distances(
    front_indices: list[int],
    fitnesses: list[float],
    model_bytes: list[int],
) -> dict[int, float]:
    """Return NSGA-II crowding distances for a Pareto front.

    Boundary candidates are assigned infinite distance so fronts keep both
    quality-leading and byte-leading topology extremes.
    """
    if not front_indices:
        return {}
    distances = {idx: 0.0 for idx in front_indices}
    if len(front_indices) <= 2:
        return {idx: float("inf") for idx in front_indices}

    objectives = (
        [float(fitnesses[idx]) for idx in front_indices],
        [float(model_bytes[idx]) for idx in front_indices],
    )
    for values in objectives:
        ordered = sorted(zip(front_indices, values), key=lambda item: (item[1], item[0]))
        distances[ordered[0][0]] = float("inf")
        distances[ordered[-1][0]] = float("inf")

        min_value = ordered[0][1]
        max_value = ordered[-1][1]
        span = max_value - min_value
        if span <= 0.0 or not math.isfinite(span):
            continue

        for pos in range(1, len(ordered) - 1):
            idx = ordered[pos][0]
            if math.isinf(distances[idx]):
                continue
            prev_value = ordered[pos - 1][1]
            next_value = ordered[pos + 1][1]
            distances[idx] += (next_value - prev_value) / span

    return distances


def sort_front_by_crowding(
    front_indices: list[int],
    fitnesses: list[float],
    model_bytes: list[int],
) -> list[int]:
    """Sort a Pareto front by crowding distance, then deterministic tie-breaks."""
    distances = crowding_distances(front_indices, fitnesses, model_bytes)
    return sorted(
        front_indices,
        key=lambda idx: (
            -distances.get(idx, 0.0),
            fitnesses[idx],
            model_bytes[idx],
            idx,
        ),
    )


def nsga2_select(
    population: list[Genome],
    fitnesses: list[float],
    model_bytes: list[int],
    count: int,
    rng: random.Random | None = None,
) -> list[Genome]:
    """Select genomes using NSGA-II non-dominated sorting.

    Fills greedily from successive fronts. Orders the last incomplete front by
    crowding distance so the frontier keeps quality and size extremes.
    """
    _ = rng
    fronts = non_dominated_sort(fitnesses, model_bytes)
    selected: list[Genome] = []
    for front_indices in fronts:
        if len(selected) + len(front_indices) <= count:
            selected.extend(population[i] for i in front_indices)
        else:
            remaining = count - len(selected)
            ordered_front = sort_front_by_crowding(front_indices, fitnesses, model_bytes)
            selected.extend(population[i] for i in ordered_front[:remaining])
            break
    return selected
