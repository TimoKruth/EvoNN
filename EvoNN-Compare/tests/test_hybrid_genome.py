from evonn_compare.hybrid.genome import HybridFamily, HybridGenome
from evonn_compare.hybrid.mutation import mutate


def test_hybrid_seed_uses_family_plus_topology() -> None:
    genome, counter = HybridGenome.create_seed(0, width=64)
    assert counter == 3
    assert len(genome.nodes) == 1
    assert len(genome.connections) == 2
    assert genome.nodes[0].family == HybridFamily.MLP


def test_hybrid_mutation_keeps_graph_nonempty() -> None:
    import random

    genome, counter = HybridGenome.create_seed(0, width=64)
    counter = mutate(genome, random.Random(42), counter)
    assert len(genome.nodes) >= 1
    assert len(genome.connections) >= 2
