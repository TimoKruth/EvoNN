import random

from stratograph.benchmarks import get_benchmark
from stratograph.genome import HierarchicalGenome
from stratograph.search import crossover_genomes, descriptor, mutate_genome, novelty_score
from stratograph.search.operators import MAX_MACRO_NODES


def _seed(name: str = "moons") -> HierarchicalGenome:
    spec = get_benchmark(name)
    return HierarchicalGenome.create_seed(
        benchmark_name=spec.name,
        task=spec.task,
        input_dim=spec.model_input_dim,
        output_dim=spec.model_output_dim,
        seed=42,
    )


def test_mutate_genome_keeps_valid_hierarchy() -> None:
    genome = _seed()
    mutated = mutate_genome(genome, rng=random.Random(42), candidate_id="mutant")
    assert mutated.genome_id == "mutant"
    assert mutated.macro_depth >= 1
    assert len(mutated.cell_library) >= 1
    assert len(mutated.macro_edges) >= len(mutated.macro_nodes)


def test_mutate_genome_without_clone_keeps_valid_hierarchy() -> None:
    genome = _seed()
    mutated = mutate_genome(genome, rng=random.Random(7), candidate_id="mutant_nc", allow_clone_mutation=False)
    assert mutated.genome_id == "mutant_nc"
    assert mutated.macro_depth >= 1
    assert len(mutated.cell_library) >= 1


def test_crossover_genome_keeps_valid_hierarchy() -> None:
    left = _seed("moons")
    right = _seed("digits")
    child = crossover_genomes(left, right, rng=random.Random(7), candidate_id="child")
    assert child.genome_id == "child"
    assert child.input_dim == left.input_dim
    assert child.output_dim == left.output_dim
    assert len(child.macro_nodes) >= 1
    assert any(edge.target == "output" for edge in child.macro_edges)
    assert any(edge.source == "input" for edge in child.macro_edges)


def test_novelty_descriptor_and_score() -> None:
    genome = _seed()
    desc = descriptor(genome)
    assert len(desc) == 4
    score = novelty_score(desc, [desc, (desc[0] + 1.0, desc[1], desc[2], desc[3])])
    assert score >= 0.0


def test_repeated_mutation_and_crossover_keep_hierarchy_bounded() -> None:
    rng = random.Random(123)
    current = _seed("digits")
    peer = _seed("moons")

    for index in range(80):
        if index % 3 == 0:
            current = crossover_genomes(current, peer, rng=rng, candidate_id=f"child_{index}")
        else:
            current = mutate_genome(current, rng=rng, candidate_id=f"mutant_{index}")
        peer = mutate_genome(peer, rng=rng, candidate_id=f"peer_{index}")

        assert len(current.macro_nodes) <= MAX_MACRO_NODES
        assert len(peer.macro_nodes) <= MAX_MACRO_NODES
        assert any(edge.source == "input" for edge in current.macro_edges)
        assert any(edge.target == "output" for edge in current.macro_edges)
