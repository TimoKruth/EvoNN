import random

from stratograph.benchmarks import get_benchmark
from stratograph.genome import HierarchicalGenome
from stratograph.pipeline.coordinator import EvaluationRecord, _select_crossover_parents, _select_diverse_elites
from stratograph.search import crossover_genomes, descriptor, mutate_genome, novelty_score


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


def test_diverse_elite_selection_keeps_descriptor_spread() -> None:
    genomes = [_seed("moons") for _ in range(4)]
    genomes = [
        genome.model_copy(update={"genome_id": f"g{index}"})
        for index, genome in enumerate(genomes)
    ]
    genomes[2] = mutate_genome(genomes[2], rng=random.Random(3), candidate_id="g2")
    genomes[3] = mutate_genome(genomes[3], rng=random.Random(9), candidate_id="g3")
    scored = [
        (
            genome,
            EvaluationRecord(
                metric_value=1.0 - index * 0.01,
                quality=1.0 - index * 0.01,
                parameter_count=10,
                train_seconds=0.1,
                architecture_summary="",
                genome_id=genome.genome_id,
                status="ok",
            ),
            float(index),
        )
        for index, genome in enumerate(genomes)
    ]

    elites = _select_diverse_elites(scored, elite_count=3)
    left, right = _select_crossover_parents(scored, elites=elites, rng=random.Random(11))

    assert len(elites) == 3
    assert len({elite.genome_id for elite in elites}) == 3
    assert left.genome_id != right.genome_id
