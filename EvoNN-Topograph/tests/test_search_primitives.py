from __future__ import annotations

import random

import numpy as np

from topograph.genome import ConnectionGene, Genome, InnovationCounter, WeightBits
from topograph.genome.genes import ExpertGene
from topograph.nn import train as train_mod
from topograph.operators import mutate as mutate_mod
from topograph.pipeline import archive as archive_mod
from topograph.pipeline import select as select_mod


def _seed_genome(seed: int = 7, num_layers: int = 3) -> Genome:
    genome = Genome.create_seed(InnovationCounter(), random.Random(seed), num_layers=num_layers)
    genome.learning_rate = 0.01
    genome.batch_size = 32
    return genome


class _FirstChoiceRng:
    def choice(self, seq):
        return seq[0]

    def randint(self, a, b):
        return a

    def random(self):
        return 0.2

    def gauss(self, mu, sigma):
        return 0.25


def test_rank_based_select_biases_toward_best():
    population = [_seed_genome(seed=i) for i in range(4)]
    selected = select_mod.rank_based_select(
        population,
        fitnesses=[0.1, 0.2, 0.3, 0.4],
        count=400,
        rng=random.Random(9),
    )

    counts = {id(genome): selected.count(genome) for genome in population}

    assert counts[id(population[0])] > counts[id(population[-1])]


def test_non_dominated_sort_and_nsga2_select_respect_fronts():
    population = [_seed_genome(seed=i) for i in range(4)]
    fronts = select_mod.non_dominated_sort(
        fitnesses=[0.1, 0.3, 0.2, 0.4],
        model_bytes=[10, 8, 12, 20],
    )

    assert fronts[0] == [0, 1]
    assert fronts[1] == [2]
    assert fronts[2] == [3]

    selected = select_mod.nsga2_select(
        population,
        fitnesses=[0.1, 0.3, 0.2, 0.4],
        model_bytes=[10, 8, 12, 20],
        count=3,
        rng=random.Random(3),
    )

    assert population[0] in selected
    assert population[1] in selected
    assert population[2] in selected


def test_compute_behavior_and_novelty_archive_roundtrip():
    genome = _seed_genome(num_layers=3)
    genome.experts = [
        ExpertGene(expert_id=0, innovation=999, width=16, activation="relu", order=1.0)
    ]
    behavior = archive_mod.compute_behavior(genome)

    assert behavior.shape == (8,)
    assert behavior[-1] == 1.0

    archive = archive_mod.NoveltyArchive(max_size=2, k=2)
    archive.add(np.zeros(8, dtype=np.float32))
    archive.add(np.ones(8, dtype=np.float32))
    archive.add(np.full(8, 2.0, dtype=np.float32))

    assert len(archive) == 2
    assert archive.compute_novelty(np.full(8, 0.5, dtype=np.float32)) > 0.0

    restored = archive_mod.NoveltyArchive.from_dict(archive.to_dict())
    assert len(restored) == 2
    assert np.allclose(restored.behaviors[-1], np.full(8, 2.0, dtype=np.float32))


def test_map_elites_and_benchmark_elite_archives_roundtrip():
    genome = _seed_genome()
    behavior = np.array([2, 2, 0, 0, 3, 4, 0.5, 0], dtype=np.float32)
    archive = archive_mod.MAPElitesArchive()

    assert archive.add(genome, behavior, fitness=0.4) is True
    assert archive.add(genome, behavior, fitness=0.5) is False
    assert archive.add(genome, behavior, fitness=0.2) is True

    sampled = archive.sample(1, random.Random(1))
    assert len(sampled) == 1
    assert sampled[0] is not genome

    restored = archive_mod.MAPElitesArchive.from_dict(archive.to_dict())
    assert len(restored) == 1

    bench_archive = archive_mod.BenchmarkEliteArchive()
    assert bench_archive.update(
        "iris",
        genome_idx=2,
        fitness=0.4,
        generation=0,
        benchmark_family="tabular",
        genome=genome,
        behavior=behavior,
        architecture_summary="3L/4C",
    ) is True
    assert bench_archive.update("iris", genome_idx=1, fitness=0.5, generation=1) is False
    assert bench_archive.update(
        "moons",
        genome_idx=3,
        fitness=0.2,
        generation=1,
        benchmark_family="tabular",
        genome=genome,
        behavior=behavior,
        architecture_summary="3L/4C",
    ) is True

    roundtrip = archive_mod.BenchmarkEliteArchive.from_dict(bench_archive.to_dict())
    assert roundtrip.get_elite_indices() == {2, 3}
    assert roundtrip.get_generation_elite_indices(0) == {2}
    assert roundtrip.elites["iris"].benchmark_family == "tabular"
    assert roundtrip.elites["iris"].genome is not None


def test_mutation_ops_preserve_copy_and_expected_structure():
    genome = _seed_genome(num_layers=3)
    genome._eval_cache = {"x": [1, 2, 3]}
    counter = InnovationCounter(100)
    rng = _FirstChoiceRng()

    copied = mutate_mod._copy_genome(genome)
    copied._eval_cache["x"].append(4)
    assert genome._eval_cache["x"] == [1, 2, 3]

    added_layer = mutate_mod.mutate_add_layer(genome, counter, rng)
    assert len(added_layer.enabled_layers) == len(genome.enabled_layers) + 1
    assert len(added_layer.enabled_connections) == len(genome.enabled_connections) + 1

    removed_layer = mutate_mod.mutate_remove_layer(genome, InnovationCounter(200), rng)
    assert len(removed_layer.enabled_layers) == len(genome.enabled_layers) - 1
    assert any(
        conn.source == 0 and conn.target == genome.layers[1].innovation
        for conn in removed_layer.enabled_connections
    )

    extra_conn = mutate_mod.mutate_add_connection(genome, InnovationCounter(300), rng)
    assert len(extra_conn.enabled_connections) == len(genome.enabled_connections) + 1


def test_mutate_weight_bits_and_learning_rate_are_bounded():
    genome = _seed_genome()
    genome.layers[0] = genome.layers[0].model_copy(update={"weight_bits": WeightBits.INT8})
    rng = _FirstChoiceRng()

    mutated_bits = mutate_mod.mutate_weight_bits(
        genome,
        rng,
        allowed_bits=[WeightBits.TERNARY, WeightBits.INT4, WeightBits.INT8],
    )
    assert mutated_bits.layers[0].weight_bits in {WeightBits.TERNARY, WeightBits.INT4}

    genome.learning_rate = 0.09
    mutated_lr = mutate_mod.mutate_learning_rate(genome, rng)
    assert 1e-5 <= mutated_lr.learning_rate <= 0.1
    assert mutated_lr.learning_rate != genome.learning_rate


def test_train_helpers_cover_metrics_percentiles_bytes_and_weight_load():
    metric = train_mod._compute_metric(
        "language_modeling",
        np.array([[0, 1]], dtype=np.int64),
        np.array([[[0.9, 0.1], [0.2, 0.8]]], dtype=np.float32),
    )
    assert metric[0] == "perplexity"
    assert metric[1] == "min"
    assert metric[2] >= 1.0
    assert metric[3] == -metric[2]

    percentiles = train_mod.compute_percentile_fitness(
        {"a": [1.0, 2.0, 2.0], "b": [3.0, 1.0, 2.0]}
    )
    assert percentiles == [0.5, 0.375, 0.625]

    genome = _seed_genome(num_layers=2)
    genome.layers[0] = genome.layers[0].model_copy(
        update={"width": 8, "weight_bits": WeightBits.INT4, "sparsity": 0.25}
    )
    genome.layers[1] = genome.layers[1].model_copy(update={"width": 4})
    genome.connections = [
        ConnectionGene(innovation=10, source=0, target=genome.layers[0].innovation),
        ConnectionGene(
            innovation=11,
            source=genome.layers[0].innovation,
            target=genome.layers[1].innovation,
        ),
        ConnectionGene(innovation=12, source=genome.layers[1].innovation, target=-1),
    ]
    assert train_mod.effective_model_bytes(genome, input_dim=6, num_classes=3) == 106

    class FakeModel:
        def __init__(self):
            self._params = {
                "dense": {
                    "weight": train_mod.mx.zeros((2, 2)),
                    "bias": train_mod.mx.zeros((2,)),
                },
                "blocks": [train_mod.mx.zeros((1,))],
            }
            self.last_update = None

        def parameters(self):
            return self._params

        def update(self, tree, strict=False):
            self.last_update = (tree, strict)

    model = FakeModel()
    loaded = train_mod.load_weight_snapshot(
        model,
        {
            "dense.weight": np.ones((2, 2), dtype=np.float32),
            "dense.bias": np.ones((3,), dtype=np.float32),
            "blocks.0": np.ones((1,), dtype=np.float32),
            "missing": np.ones((1,), dtype=np.float32),
        },
    )

    assert loaded == 2
    assert model.last_update is not None
    tree, strict = model.last_update
    assert strict is False
    assert tree["dense"]["weight"].shape == (2, 2)
    assert tree["blocks"][0].shape == (1,)
