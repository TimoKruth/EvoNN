"""NEAT-style crossover for topology-based genomes."""

from __future__ import annotations

import random

from topograph.genome import ConnectionGene, Genome, LayerGene


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
    disabled_gene_chance: float = 0.25,
) -> Genome:
    """Create a child genome from two parents using NEAT alignment."""
    a_fit = parent_a.fitness if parent_a.fitness is not None else float("inf")
    b_fit = parent_b.fitness if parent_b.fitness is not None else float("inf")

    equal_fitness = a_fit == b_fit
    a_fitter = a_fit <= b_fit
    b_fitter = b_fit <= a_fit

    child_layers = _align_genes(
        parent_a.layers, parent_b.layers, a_fitter, b_fitter, equal_fitness,
        rng, disabled_gene_chance,
    )
    child_connections = _align_genes(
        parent_a.connections, parent_b.connections, a_fitter, b_fitter,
        equal_fitness, rng, disabled_gene_chance,
    )

    child = Genome(layers=child_layers, connections=child_connections)

    if parent_a.conv_layers or parent_b.conv_layers:
        child.conv_layers = _align_genes(
            parent_a.conv_layers, parent_b.conv_layers, a_fitter, b_fitter,
            equal_fitness, rng, disabled_gene_chance,
        )

    return child


def _align_genes(
    genes_a: list[LayerGene | ConnectionGene],
    genes_b: list[LayerGene | ConnectionGene],
    a_fitter: bool,
    b_fitter: bool,
    equal_fitness: bool,
    rng: random.Random,
    disabled_gene_chance: float,
) -> list:
    """Align genes by innovation number following NEAT rules."""
    map_a = {g.innovation: g for g in genes_a}
    map_b = {g.innovation: g for g in genes_b}

    child: list = []
    for inn in sorted(set(map_a) | set(map_b)):
        in_a, in_b = inn in map_a, inn in map_b

        if in_a and in_b:
            gene = rng.choice([map_a[inn], map_b[inn]])
            if not map_a[inn].enabled or not map_b[inn].enabled:
                gene = gene.model_copy(
                    update={"enabled": rng.random() >= disabled_gene_chance}
                )
            child.append(gene)
        elif in_a:
            if a_fitter or (equal_fitness and rng.random() < 0.5):
                child.append(map_a[inn])
        elif in_b:
            if b_fitter or (equal_fitness and rng.random() < 0.5):
                child.append(map_b[inn])

    return child
