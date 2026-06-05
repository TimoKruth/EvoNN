import random

from stratograph.benchmarks import get_benchmark
from stratograph.genome import HierarchicalGenome
from stratograph.genome.models import MacroNodeGene
from stratograph.pipeline.coordinator import (
    _benchmark_profile_key,
    _next_population,
    _offspring_mutation_modes,
    _profile_survival_bonus,
    _select_parent_pool,
)
from stratograph.pipeline.evaluator import EvaluationRecord
from stratograph.search import crossover_genomes, descriptor, mutate_genome, novelty_score
from stratograph.search.operators import MAX_MACRO_NODES, TASK_MOTIFS


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


def test_regression_motif_mutation_is_available() -> None:
    genome = _seed("diabetes")
    mutated = mutate_genome(
        genome,
        rng=random.Random(9),
        candidate_id="regression_motif",
        preferred_modes=("motif_rewrite",),
    )

    assert "regression" in TASK_MOTIFS
    assert mutated.genome_id == "regression_motif"
    assert mutated.task == "regression"
    assert mutated.macro_depth >= 1


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

    def fake_mutate(parent, *, rng, candidate_id, allow_clone_mutation=True, motif_bias=True, motif_task=None, preferred_modes=None):
        selected_parents.append(("mutate", parent.genome_id))
        return parent.model_copy(update={"genome_id": candidate_id})

    def fake_crossover(
        left,
        right,
        *,
        rng,
        candidate_id,
        allow_clone_mutation=True,
        motif_bias=True,
        motif_task=None,
        preferred_mutation_modes=None,
    ):
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


def test_shared_parent_pool_keeps_benchmark_reuse_and_niche_elites() -> None:
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
    shared_cell = next(iter(base.cell_library.values()))
    medium_reuse_cells = dict(base.cell_library)
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
        f"cell_low_{index}": shared_cell.model_copy(update={"cell_id": f"cell_low_{index}", "shared": False}, deep=True)
        for index, _ in enumerate(base.macro_nodes)
    }
    benchmark_leader = base.model_copy(
        update={"genome_id": "benchmark_leader", "macro_nodes": low_reuse_nodes, "cell_library": low_reuse_cells}
    )

    scored = [
        (benchmark_leader, EvaluationRecord(0.98, 0.98, 10, 1.0, "", benchmark_leader.genome_id, "ok"), 0.01),
        (medium_reuse, EvaluationRecord(0.93, 0.93, 10, 1.0, "", medium_reuse.genome_id, "ok"), 0.50),
        (high_reuse, EvaluationRecord(0.80, 0.80, 10, 1.0, "", high_reuse.genome_id, "ok"), 0.20),
    ]

    parents = _select_parent_pool(
        scored,
        population_size=4,
        architecture_mode="two_level_shared",
        profile_key="tabular",
    )

    assert [parent.genome_id for parent in parents] == ["benchmark_leader", "high_reuse", "medium_reuse"]


def test_profile_key_and_mutation_modes_prioritize_image_specific_search() -> None:
    profile = _benchmark_profile_key(
        benchmark_name="digits_image",
        task="classification",
        input_dim=64,
        output_dim=10,
    )

    assert profile == "image"
    assert _offspring_mutation_modes(
        0,
        task="classification",
        profile_key=profile,
        architecture_mode="two_level_shared",
        allow_clone_mutation=True,
        motif_bias=True,
    ) == ("motif_rewrite", "specialize_cell")


def test_tabular_profile_uses_motif_rewrite_and_survival_bonus() -> None:
    genome = _seed("openml_gas_sensor")
    profile = _benchmark_profile_key(
        benchmark_name="openml_gas_sensor",
        task=genome.task,
        input_dim=genome.input_dim,
        output_dim=genome.output_dim,
    )

    assert "tabular" in TASK_MOTIFS
    assert profile == "tabular"
    assert _offspring_mutation_modes(
        0,
        task=genome.task,
        profile_key=profile,
        architecture_mode="two_level_shared",
        allow_clone_mutation=True,
        motif_bias=True,
    ) == ("motif_rewrite", "activation")
    assert _profile_survival_bonus(genome, profile) > 0.0


def test_lm_profile_key_mutation_modes_and_survival_bonus_are_sequence_aware() -> None:
    genome = _seed("tiny_lm_synthetic")
    profile = _benchmark_profile_key(
        benchmark_name="tiny_lm_synthetic",
        task="language_modeling",
        input_dim=genome.input_dim,
        output_dim=genome.output_dim,
    )

    assert profile == "language_modeling"
    assert _offspring_mutation_modes(
        1,
        task="language_modeling",
        profile_key=profile,
        architecture_mode="two_level_shared",
        allow_clone_mutation=True,
        motif_bias=True,
    ) == ("add_skip_edge", "rewire_macro")
    assert _profile_survival_bonus(genome, profile) > 0.0
