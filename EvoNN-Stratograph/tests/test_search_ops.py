import random

from stratograph.benchmarks import get_benchmark
from stratograph.genome import HierarchicalGenome
from stratograph.genome.models import MacroNodeGene
from stratograph.pipeline.coordinator import _next_population
from stratograph.pipeline.evaluator import EvaluationRecord
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



def test_next_population_shared_mode_keeps_high_reuse_leader_in_parent_pool(monkeypatch) -> None:
    base = _seed("moons")
    high_reuse = base.model_copy(update={"genome_id": "high_reuse"})

    medium_reuse_nodes = [node.model_copy() for node in base.macro_nodes]
    medium_reuse_nodes[-1] = MacroNodeGene(
        node_id=medium_reuse_nodes[-1].node_id,
        cell_id="cell_alt",
        input_width=medium_reuse_nodes[-1].input_width,
        output_width=medium_reuse_nodes[-1].output_width,
        role=medium_reuse_nodes[-1].role,
    )
    medium_reuse_cells = dict(base.cell_library)
    shared_cell = next(iter(base.cell_library.values()))
    medium_reuse_cells["cell_alt"] = shared_cell.model_copy(update={"cell_id": "cell_alt", "shared": False}, deep=True)
    medium_reuse = base.model_copy(
        update={"genome_id": "medium_reuse", "macro_nodes": medium_reuse_nodes, "cell_library": medium_reuse_cells}
    )

    low_reuse_nodes = [
        MacroNodeGene(
            node_id=node.node_id,
            cell_id=f"cell_low_{index}",
            input_width=node.input_width,
            output_width=node.output_width,
            role=node.role,
        )
        for index, node in enumerate(base.macro_nodes)
    ]
    low_reuse_cells = {
        f"cell_low_{index}": cell.model_copy(update={"cell_id": f"cell_low_{index}", "shared": False}, deep=True)
        for index, cell in enumerate([next(iter(base.cell_library.values())) for _ in base.macro_nodes])
    }
    low_reuse = base.model_copy(
        update={"genome_id": "low_reuse", "macro_nodes": low_reuse_nodes, "cell_library": low_reuse_cells}
    )

    evaluated = [
        (low_reuse, EvaluationRecord(0.95, 0.95, 10, 1.0, "", low_reuse.genome_id, "ok"), 0.01),
        (medium_reuse, EvaluationRecord(0.93, 0.93, 10, 1.0, "", medium_reuse.genome_id, "ok"), 0.02),
        (high_reuse, EvaluationRecord(0.80, 0.80, 10, 1.0, "", high_reuse.genome_id, "ok"), 0.20),
    ]

    selected_parents: list[tuple[str, ...]] = []

    def fake_mutate(parent, *, rng, candidate_id, allow_clone_mutation=True, motif_bias=True):
        selected_parents.append(("mutate", parent.genome_id))
        return parent.model_copy(update={"genome_id": candidate_id})

    def fake_crossover(left, right, *, rng, candidate_id, allow_clone_mutation=True, motif_bias=True):
        selected_parents.append(("crossover", left.genome_id, right.genome_id))
        return left.model_copy(update={"genome_id": candidate_id})

    monkeypatch.setattr("stratograph.pipeline.coordinator.mutate_genome", fake_mutate)
    monkeypatch.setattr("stratograph.pipeline.coordinator.crossover_genomes", fake_crossover)

    _next_population(
        evaluated=evaluated,
        benchmark_name="moons",
        task=base.task,
        input_dim=base.input_dim,
        output_dim=base.output_dim,
        seed=7,
        generation=0,
        population_size=4,
        architecture_mode="two_level_shared",
        allow_clone_mutation=True,
        motif_bias=True,
        trained_states={genome.genome_id: None for genome, _, _ in evaluated},
    )

    assert any("high_reuse" in parents for parents in selected_parents)
