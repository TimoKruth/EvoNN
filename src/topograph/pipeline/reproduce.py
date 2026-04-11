"""Reproduction stage: crossover, mutation, and next-generation assembly."""

from __future__ import annotations

import random

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
from topograph.pipeline.schedule import MutationScheduler
from topograph.pipeline.select import rank_based_select


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


def reproduce(
    state: GenerationState,
    config: RunConfig,
    innovation_counter: InnovationCounter,
    scheduler: MutationScheduler,
    rng: random.Random,
) -> tuple[GenerationState, list[tuple[int, list[str]]]]:
    """Create next generation from selected parents.

    Returns (updated_state, applied_ops) where applied_ops is a list of
    (genome_index_in_new_pop, [operator_names]) for scheduler feedback.
    """
    rates = scheduler.get_rates(state.generation, config.evolution.num_generations)

    elite_count = config.evolution.elite_count
    pop_size = config.evolution.population_size
    crossover_ratio = config.evolution.crossover_ratio

    # Elites pass unchanged (lowest fitness = best)
    sorted_indices = sorted(range(len(state.fitnesses)), key=lambda i: state.fitnesses[i])
    elites = [_clone_genome(state.population[i]) for i in sorted_indices[:elite_count]]

    offspring: list[Genome] = []
    applied_ops: list[tuple[int, list[str]]] = []

    while len(elites) + len(offspring) < pop_size:
        if rng.random() < crossover_ratio and len(state.population) >= 2:
            parents = rank_based_select(state.population, state.fitnesses, 2, rng)
            child = crossover(parents[0], parents[1], rng)
        else:
            [parent] = rank_based_select(state.population, state.fitnesses, 1, rng)
            child = _clone_genome(parent)

        # Apply mutations
        child, ops = _apply_mutations(child, rates, innovation_counter, rng, config)

        offspring_idx = len(elites) + len(offspring)
        offspring.append(child)
        if ops:
            applied_ops.append((offspring_idx, ops))

    new_population = elites + offspring
    state.population = new_population
    # Clear fitnesses for next generation (elites will be re-evaluated)
    state.fitnesses = []
    state.model_bytes = []
    state.behaviors = []
    return state, applied_ops


def _clone_genome(genome: Genome) -> Genome:
    """Deep-copy a genome using the mutate module's copy helper."""
    return _copy_genome(genome)


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
