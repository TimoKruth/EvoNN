from __future__ import annotations

from random import Random

from evonn_primordia.search_state import CandidateSeed, EliteArchive


class _Genome:
    def __init__(self, genome_id: str) -> None:
        self.genome_id = genome_id


def test_elite_archive_retains_highest_search_score_records() -> None:
    archive = EliteArchive(0.4)
    archive.update({"genome_id": "a", "primitive_family": "mlp", "search_score": 0.5})
    archive.update({"genome_id": "b", "primitive_family": "mlp", "search_score": 0.9})
    archive.update({"genome_id": "c", "primitive_family": "embedding", "search_score": 0.7})

    elites = archive.elites(total_budget=5)

    assert elites[0]["genome_id"] == "b"
    assert {row["genome_id"] for row in elites} >= {"b", "c"}


def test_elite_archive_sampling_keeps_family_exploration_floor() -> None:
    archive = EliteArchive(0.5)
    archive.update({"genome_id": "a", "primitive_family": "mlp", "search_score": 0.8})
    archive.update({"genome_id": "b", "primitive_family": "embedding", "search_score": 0.7})
    archive.update({"genome_id": "c", "primitive_family": "mlp", "search_score": 0.6})

    parents = archive.sample_parent_records(count=2, total_budget=6, rng=Random(42), family_exploration_floor=1)

    assert len(parents) == 2
    assert {row["primitive_family"] for row in parents} == {"mlp", "embedding"}


def test_elite_archive_sampling_prioritizes_benchmark_leader_then_diverse_family_leaders() -> None:
    archive = EliteArchive(0.8)
    archive.update({"genome_id": "mlp-best", "primitive_family": "mlp", "search_score": 0.95, "generation": 2, "novelty_score": 0.2})
    archive.update({"genome_id": "sparse-best", "primitive_family": "sparse_mlp", "search_score": 0.91, "generation": 1, "novelty_score": 0.8})
    archive.update({"genome_id": "embed-best", "primitive_family": "embedding", "search_score": 0.89, "generation": 3, "novelty_score": 1.0})
    archive.update({"genome_id": "mlp-old", "primitive_family": "mlp", "search_score": 0.75, "generation": 0, "novelty_score": 0.1})

    parents = archive.sample_parent_records(count=3, total_budget=6, rng=Random(7), family_exploration_floor=1)

    assert [row["genome_id"] for row in parents] == ["mlp-best", "sparse-best", "embed-best"]


def test_elite_archive_rank_weights_preserve_selection_pressure() -> None:
    records = [
        {"genome_id": "best", "primitive_family": "mlp", "search_score": 0.95},
        {"genome_id": "middle", "primitive_family": "conv2d", "search_score": 0.25},
        {"genome_id": "weak", "primitive_family": "lite_conv2d", "search_score": 0.1},
    ]

    weights = EliteArchive._selection_weights(records)

    assert weights[0] > weights[1] > weights[2]
    assert weights[0] / weights[2] >= 5.0


def test_elite_archive_can_reuse_best_parent_when_family_floor_is_disabled() -> None:
    archive = EliteArchive(1.0)
    archive.update({"genome_id": "mlp-best", "primitive_family": "mlp", "search_score": 0.95})
    archive.update({"genome_id": "conv-weak", "primitive_family": "conv2d", "search_score": 0.25})
    archive.update({"genome_id": "lite-weak", "primitive_family": "lite_conv2d", "search_score": 0.1})

    parents = archive.sample_parent_records(
        count=3,
        total_budget=6,
        rng=Random(1),
        family_exploration_floor=0,
        allow_duplicate_parent_sampling=True,
    )

    assert [row["genome_id"] for row in parents].count("mlp-best") >= 2


def test_candidate_seed_carries_lineage_fields() -> None:
    seed = CandidateSeed(genome=_Genome("g1"), generation=2, parent_genome_id="parent", mutation_operator="width")

    assert seed.generation == 2
    assert seed.parent_genome_id == "parent"
    assert seed.mutation_operator == "width"
