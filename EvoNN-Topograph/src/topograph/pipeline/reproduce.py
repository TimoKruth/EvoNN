"""Reproduction stage: crossover, mutation, and next-generation assembly."""

from __future__ import annotations

import random
from dataclasses import dataclass

from topograph.config import RunConfig
from topograph.genome.genome import Genome, InnovationCounter
from topograph.operators.crossover import crossover
from topograph.operators.mutate import (
    _copy_genome,
    mutate_activation,
    mutate_activation_bits,
    mutate_add_connection,
    mutate_add_layer,
    mutate_add_residual,
    mutate_batch_size,
    mutate_learning_rate,
    mutate_operator_type,
    mutate_remove_connection,
    mutate_remove_layer,
    mutate_sparsity,
    mutate_weight_bits,
    mutate_width,
)
from topograph.pipeline.evaluate import GenerationState
from topograph.pipeline.archive import compute_behavior
from topograph.pipeline.schedule import MutationScheduler
from topograph.pipeline.select import non_dominated_sort, rank_based_select


# Mutation dispatch table: operator name -> (function, needs_counter)
_MUTATION_TABLE: dict[str, tuple] = {
    "width": (mutate_width, False),
    "activation": (mutate_activation, False),
    "add_layer": (mutate_add_layer, True),
    "remove_layer": (mutate_remove_layer, True),
    "add_connection": (mutate_add_connection, True),
    "remove_connection": (mutate_remove_connection, False),
    "add_residual": (mutate_add_residual, True),
    "weight_bits": (mutate_weight_bits, False),
    "activation_bits": (mutate_activation_bits, False),
    "sparsity": (mutate_sparsity, False),
    "operator_type": (mutate_operator_type, False),
}


@dataclass
class PendingMutationOutcome:
    genome_idx: int
    baseline_fitness: float
    operators: list[str]


def reproduce(
    state: GenerationState,
    config: RunConfig,
    innovation_counter: InnovationCounter,
    scheduler: MutationScheduler,
    rng: random.Random,
    protected_indices: set[int] | None = None,
) -> tuple[GenerationState, list[PendingMutationOutcome]]:
    """Create next generation from selected parents.

    Returns (updated_state, pending_outcomes) where pending_outcomes records
    parent baselines for the next generation's scheduler feedback.
    """
    rates = scheduler.get_rates(state.generation, config.evolution.num_generations)

    elite_count = config.evolution.elite_count
    pop_size = config.evolution.population_size
    crossover_ratio = config.evolution.crossover_ratio

    ranked_indices = _ranked_indices(state, config)
    protected_indices = protected_indices or set()
    selection_population = [state.population[i] for i in ranked_indices]
    selection_fitnesses = list(range(len(selection_population)))
    elites = _select_survivors(
        state,
        ranked_indices=ranked_indices,
        elite_count=elite_count,
        protected_indices=protected_indices,
        pop_size=pop_size,
    )

    for genome in elites:
        _clear_metrics(genome, drop_eval_cache=False)

    offspring: list[Genome] = []
    pending_outcomes: list[PendingMutationOutcome] = []

    while len(elites) + len(offspring) < pop_size:
        used_crossover = False
        if rng.random() < crossover_ratio and len(selection_population) >= 2:
            parents = rank_based_select(selection_population, selection_fitnesses, 2, rng)
            child = crossover(parents[0], parents[1], rng)
            used_crossover = True
            baseline_fitness = min(
                _fitness_or_inf(parents[0]),
                _fitness_or_inf(parents[1]),
            )
        else:
            [parent] = rank_based_select(selection_population, selection_fitnesses, 1, rng)
            child = _clone_genome(parent)
            baseline_fitness = _fitness_or_inf(parent)

        # Apply mutations
        child, ops = _apply_mutations(child, rates, innovation_counter, rng, config)
        _clear_metrics(child, drop_eval_cache=used_crossover or bool(ops))

        offspring_idx = len(elites) + len(offspring)
        offspring.append(child)
        if ops:
            pending_outcomes.append(
                PendingMutationOutcome(
                    genome_idx=offspring_idx,
                    baseline_fitness=baseline_fitness,
                    operators=ops,
                )
            )

    new_population = elites + offspring
    state.population = new_population
    # Clear fitnesses for next generation (elites will be re-evaluated)
    state.fitnesses = []
    state.model_bytes = []
    state.behaviors = []
    state.raw_losses = {}
    return state, pending_outcomes


def _clone_genome(genome: Genome) -> Genome:
    """Deep-copy a genome using the mutate module's copy helper."""
    return _copy_genome(genome)


def _clear_metrics(genome: Genome, drop_eval_cache: bool) -> None:
    genome.fitness = None
    genome.param_count = 0
    genome.model_bytes = 0
    if hasattr(genome, "_last_eval_reused"):
        delattr(genome, "_last_eval_reused")
    if drop_eval_cache and hasattr(genome, "_eval_cache"):
        delattr(genome, "_eval_cache")


def _fitness_or_inf(genome: Genome) -> float:
    return float(genome.fitness) if genome.fitness is not None else float("inf")


def _ranked_indices(state: GenerationState, config: RunConfig) -> list[int]:
    if config.objectives and state.model_bytes and len(state.model_bytes) == len(state.population):
        fronts = non_dominated_sort(state.fitnesses, state.model_bytes)
        ranked: list[int] = []
        for front in fronts:
            ranked.extend(
                sorted(front, key=lambda idx: (state.fitnesses[idx], state.model_bytes[idx], idx))
            )
        return ranked
    return sorted(range(len(state.fitnesses)), key=lambda idx: (state.fitnesses[idx], idx))


def _select_survivors(
    state: GenerationState,
    *,
    ranked_indices: list[int],
    elite_count: int,
    protected_indices: set[int],
    pop_size: int,
) -> list[Genome]:
    survivors: list[Genome] = []
    added: set[int] = set()

    for idx in ranked_indices[:elite_count]:
        survivors.append(_clone_genome(state.population[idx]))
        added.add(idx)

    for idx in ranked_indices:
        if len(survivors) >= pop_size:
            break
        if idx in protected_indices and idx not in added:
            survivors.append(_clone_genome(state.population[idx]))
            added.add(idx)

    diversity_target = min(
        pop_size,
        max(len(survivors), elite_count) + max(1, pop_size // 8),
    )
    for idx in _topology_diverse_indices(state, ranked_indices):
        if len(survivors) >= diversity_target:
            break
        if idx not in added:
            survivors.append(_clone_genome(state.population[idx]))
            added.add(idx)

    return survivors


def _topology_diverse_indices(state: GenerationState, ranked_indices: list[int]) -> list[int]:
    if not state.population:
        return []
    candidates: list[tuple[float, int]] = []
    for idx in ranked_indices:
        genome = state.population[idx]
        behavior = compute_behavior(genome)
        topology_score = (
            float(behavior[0]) * 0.20
            + float(behavior[2]) * 0.25
            + float(behavior[6]) * 0.20
            + float(len({layer.operator for layer in genome.enabled_layers})) * 0.15
        )
        if state.fitnesses and idx < len(state.fitnesses) and state.fitnesses[idx] != float("inf"):
            topology_score += 1.0 / (1.0 + max(0.0, float(state.fitnesses[idx])))
        candidates.append((topology_score, idx))
    return [idx for _, idx in sorted(candidates, reverse=True)]


def _apply_mutations(
    genome: Genome,
    rates: dict[str, float],
    innovation_counter: InnovationCounter,
    rng: random.Random,
    config: RunConfig,
) -> tuple[Genome, list[str]]:
    """Apply probabilistic mutations. Returns (mutated_genome, applied_operator_names)."""
    applied: list[str] = []

    # Quantization phase constraints
    allowed_weight_bits = None
    allowed_activation_bits = None
    if config.quantization_schedule:
        from topograph.genome.genes import ActivationBits, WeightBits

        gen = getattr(genome, "_current_gen", None)  # set by coordinator if needed
        for phase in config.quantization_schedule:
            start, end = phase.generations
            if end is None:
                end = config.evolution.num_generations
            if start <= (gen or 0) < end:
                allowed_weight_bits = [WeightBits(b) for b in phase.allowed_weight_bits]
                allowed_activation_bits = [ActivationBits(b) for b in phase.allowed_activation_bits]
                break

    for op_name, prob in rates.items():
        if rng.random() < prob:
            genome = _dispatch_mutation(
                genome, op_name, innovation_counter, rng,
                allowed_weight_bits=allowed_weight_bits,
                allowed_activation_bits=allowed_activation_bits,
            )
            applied.append(op_name)

    if not applied and rng.random() < 0.5:
        op_name = rng.choice(["add_connection", "add_residual", "operator_type", "width"])
        genome = _dispatch_mutation(
            genome,
            op_name,
            innovation_counter,
            rng,
            allowed_weight_bits=allowed_weight_bits,
            allowed_activation_bits=allowed_activation_bits,
        )
        applied.append(op_name)

    # Hyperparameter mutations (independent of scheduled rates)
    if genome.learning_rate is not None and rng.random() < 0.3:
        genome = mutate_learning_rate(genome, rng)
        applied.append("learning_rate")

    if rng.random() < 0.2:
        genome = mutate_batch_size(genome, rng)
        applied.append("batch_size")

    return genome, applied


def _dispatch_mutation(
    genome: Genome,
    op_name: str,
    innovation_counter: InnovationCounter,
    rng: random.Random,
    allowed_weight_bits=None,
    allowed_activation_bits=None,
) -> Genome:
    """Dispatch to specific mutation function by name."""
    entry = _MUTATION_TABLE.get(op_name)
    if entry is None:
        return genome

    fn, needs_counter = entry

    # Special cases with extra kwargs
    if op_name == "weight_bits":
        return fn(genome, rng, allowed_bits=allowed_weight_bits)
    if op_name == "activation_bits":
        return fn(genome, rng, allowed_bits=allowed_activation_bits)
    if op_name == "remove_connection":
        return fn(genome, rng)

    # Standard dispatch
    if needs_counter:
        return fn(genome, innovation_counter, rng)
    return fn(genome, rng)
