"""Selection operators for evolutionary search."""

from __future__ import annotations

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


def nsga2_select(
    population: list[Genome],
    fitnesses: list[float],
    model_bytes: list[int],
    count: int,
    rng: random.Random | None = None,
) -> list[Genome]:
    """Select genomes using NSGA-II non-dominated sorting.

    Fills greedily from successive fronts. Shuffles the last incomplete front
    and takes the remainder needed.
    """
    fronts = non_dominated_sort(fitnesses, model_bytes)
    selected: list[Genome] = []
    for front_indices in fronts:
        if len(selected) + len(front_indices) <= count:
            selected.extend(population[i] for i in front_indices)
        else:
            remaining = count - len(selected)
            if rng is not None:
                rng.shuffle(front_indices)
            selected.extend(population[i] for i in front_indices[:remaining])
            break
    return selected
