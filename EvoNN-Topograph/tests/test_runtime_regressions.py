from __future__ import annotations

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from topograph.cache import WeightCache
from topograph.config import RunConfig
from topograph.genome.codec import genome_to_dict
from topograph.genome.genome import Genome, InnovationCounter
from topograph.pipeline import coordinator as coordinator_mod
from topograph.pipeline.evaluate import GenerationState, _aggregate_pool_fitness, _make_evaluation_plan
from topograph.pipeline.reproduce import reproduce
from topograph.pipeline.schedule import MutationScheduler
from topograph.storage import RunStore


def _seed_genome(seed: int) -> Genome:
    genome = Genome.create_seed(InnovationCounter(), random.Random(seed), num_layers=3)
    genome.learning_rate = 0.01
    genome.batch_size = 16
    return genome


def test_run_evolution_interrupt_saves_resume_snapshot(tmp_path: Path, monkeypatch):
    cfg = RunConfig.model_validate(
        {
            "benchmark": "moons",
            "evolution": {"population_size": 1, "num_generations": 1, "elite_count": 1},
            "benchmark_pool": {
                "benchmarks": ["bench_a", "bench_b", "bench_c"],
                "sample_k": 2,
                "rotation_interval": 3,
            },
            "benchmark_elite_archive": True,
        }
    )
    run_dir = tmp_path / "interrupt-run"

    observed: list[list[str]] = []

    monkeypatch.setattr(
        coordinator_mod,
        "get_benchmark",
        lambda name: SimpleNamespace(
            name=name, task="classification", source="sklearn", input_dim=2, num_classes=2
        ),
    )

    def fake_evaluate_pool(**kwargs):
        observed.append([spec.name for spec in kwargs["benchmark_specs"]])
        raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator_mod, "evaluate_pool", fake_evaluate_pool)

    with pytest.raises(KeyboardInterrupt):
        coordinator_mod.run_evolution(cfg, run_dir=str(run_dir))

    with RunStore(run_dir / "metrics.duckdb") as store:
        snapshot = store.load_run_state("current")

    assert observed and len(observed[0]) == 2
    assert snapshot["next_generation"] == 0
    assert snapshot["completed"] is False
    assert snapshot["pool_state"]["current_sample"] == observed[0]
    assert snapshot["pool_state"]["rotation_counter"] == 1
    assert snapshot["pool_state"]["family_stage_history"][0]["active_family"] == "tabular"


def test_run_evolution_resume_preserves_pool_rotation_state(tmp_path: Path, monkeypatch):
    cfg = RunConfig.model_validate(
        {
            "benchmark": "moons",
            "evolution": {"population_size": 1, "num_generations": 1, "elite_count": 1},
            "benchmark_pool": {
                "benchmarks": ["bench_a", "bench_b", "bench_c"],
                "sample_k": 2,
                "rotation_interval": 3,
            },
            "benchmark_elite_archive": True,
        }
    )
    genome = _seed_genome(7)
    run_dir = tmp_path / "resume-run"
    run_dir.mkdir()

    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run("current", cfg.model_dump(mode="json"))
        store.save_run_state(
            "current",
            {
                "next_generation": 0,
                "population": [genome_to_dict(genome)],
                "innovation_counter": 9,
                "fitness_history": [],
                "scheduler": {"stats": {}},
                "pool_state": {
                    "current_sample": ["bench_b", "bench_c"],
                    "rotation_counter": 1,
                    "benchmark_best_fitness": {},
                },
                "pending_outcomes": [],
                "elapsed_seconds": 0.5,
                "total_evaluations": 0,
                "novelty_score_sum": 0.0,
                "novelty_score_count": 0,
                "novelty_score_max": 0.0,
                "map_elites_insertions": 0,
                "completed": False,
            },
        )

    observed: list[list[str]] = []

    monkeypatch.setattr(
        coordinator_mod,
        "get_benchmark",
        lambda name: SimpleNamespace(
            name=name, task="classification", source="sklearn", input_dim=2, num_classes=2
        ),
    )

    def fake_evaluate_pool(**kwargs):
        state = kwargs["state"]
        specs = kwargs["benchmark_specs"]
        observed.append([spec.name for spec in specs])
        state.fitnesses = [0.2]
        state.model_bytes = [16]
        state.behaviors = [np.zeros(8, dtype=np.float32)]
        state.raw_losses = {spec.name: [0.2 + idx * 0.1] for idx, spec in enumerate(specs)}
        state.benchmark_results = []
        state.benchmark_timings = [
            {
                "benchmark_order": idx,
                "benchmark_total": len(specs),
                "benchmark_name": spec.name,
                "task": spec.task,
                "data_load_seconds": 0.01,
                "evaluation_seconds": 0.02,
                "total_seconds": 0.03,
                "trained_count": 1,
                "reused_count": 0,
                "failed_count": 0,
                "requested_worker_count": 1,
                "resolved_worker_count": 1,
                "data_cache_hits": 0,
                "data_cache_misses": 1,
                "worker_clamp_reason": "sequential",
            }
            for idx, spec in enumerate(specs)
        ]
        state.total_evaluations += len(specs)
        return state

    monkeypatch.setattr(coordinator_mod, "evaluate_pool", fake_evaluate_pool)
    monkeypatch.setattr(coordinator_mod, "score", lambda state, config: state)

    final_state = coordinator_mod.run_evolution(cfg, run_dir=str(run_dir), resume=True)

    with RunStore(run_dir / "metrics.duckdb") as store:
        snapshot = store.load_run_state("current")

    assert observed == [["bench_b", "bench_c"]]
    assert final_state.total_evaluations == 2
    assert snapshot["completed"] is True
    assert snapshot["pool_state"]["current_sample"] == ["bench_b", "bench_c"]
    assert snapshot["pool_state"]["rotation_counter"] == 2
    assert (run_dir / "benchmark_elites.json").exists()
    assert (run_dir / "topology_atlas_summary.json").exists()


def test_run_evolution_resume_preserves_existing_primordia_seeding_metadata(tmp_path: Path, monkeypatch):
    cfg = RunConfig(
        benchmark="moons",
        evolution={"population_size": 1, "num_generations": 2, "elite_count": 1},
        benchmark_elite_archive=False,
    )
    genome = _seed_genome(13)
    run_dir = tmp_path / "seeded-resume-run"
    run_dir.mkdir()

    original_seeding = {
        "seed_path": "/tmp/seed_candidates.json",
        "target_family": "tabular",
        "selected_family": "sparse_mlp",
        "selected_rank": 1,
        "representative_architecture_summary": "sparse_mlp[64x64]",
        "representative_genome_id": "sparse-1",
    }

    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run("current", cfg.model_dump(mode="json"))
        store.save_budget_metadata("current", {"primordia_seeding": original_seeding})
        store.save_run_state(
            "current",
            {
                "next_generation": 1,
                "population": [genome_to_dict(genome)],
                "innovation_counter": 9,
                "fitness_history": [0.1],
                "scheduler": {"stats": {}},
                "pool_state": {
                    "current_sample": [],
                    "rotation_counter": 0,
                    "benchmark_best_fitness": {},
                },
                "pending_outcomes": [],
                "elapsed_seconds": 0.5,
                "total_evaluations": 1,
                "novelty_score_sum": 0.0,
                "novelty_score_count": 0,
                "novelty_score_max": 0.0,
                "map_elites_insertions": 0,
                "completed": False,
            },
        )

    def fake_evaluate(
        state,
        config,
        benchmark_spec,
        cache=None,
        multi_fidelity_schedule=None,
        data_cache=None,
        evaluation_memo=None,
        parallel_eval=None,
        progress_callback=None,
    ):
        state.fitnesses = [0.2]
        state.model_bytes = [16]
        state.behaviors = [np.zeros(8, dtype=np.float32)]
        state.raw_losses = {"moons": [0.2]}
        state.benchmark_results = []
        state.total_evaluations += 1
        return state

    monkeypatch.setattr(coordinator_mod, "evaluate", fake_evaluate)
    monkeypatch.setattr(coordinator_mod, "score", lambda state, config: state)

    coordinator_mod.run_evolution(
        cfg,
        benchmark_spec=SimpleNamespace(name="moons", task="classification", input_dim=2, num_classes=2),
        run_dir=str(run_dir),
        resume=True,
    )

    with RunStore(run_dir / "metrics.duckdb") as store:
        budget_meta = store.load_budget_metadata("current")

    assert budget_meta["primordia_seeding"] == original_seeding


def test_run_evolution_resume_ignores_non_mapping_primordia_seeding_metadata(tmp_path: Path, monkeypatch):
    cfg = RunConfig(
        benchmark="moons",
        evolution={"population_size": 1, "num_generations": 2, "elite_count": 1},
        benchmark_elite_archive=False,
    )
    genome = _seed_genome(17)
    run_dir = tmp_path / "bad-seeding-resume-run"
    run_dir.mkdir()

    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run("current", cfg.model_dump(mode="json"))
        store.save_budget_metadata("current", {"primordia_seeding": "not-a-dict"})
        store.save_run_state(
            "current",
            {
                "next_generation": 1,
                "population": [genome_to_dict(genome)],
                "innovation_counter": 9,
                "fitness_history": [0.1],
                "scheduler": {"stats": {}},
                "pool_state": {
                    "current_sample": [],
                    "rotation_counter": 0,
                    "benchmark_best_fitness": {},
                },
                "pending_outcomes": [],
                "elapsed_seconds": 0.5,
                "total_evaluations": 1,
                "novelty_score_sum": 0.0,
                "novelty_score_count": 0,
                "novelty_score_max": 0.0,
                "map_elites_insertions": 0,
                "completed": False,
            },
        )

    def fake_evaluate(
        state,
        config,
        benchmark_spec,
        cache=None,
        multi_fidelity_schedule=None,
        data_cache=None,
        evaluation_memo=None,
        parallel_eval=None,
        progress_callback=None,
    ):
        state.fitnesses = [0.2]
        state.model_bytes = [16]
        state.behaviors = [np.zeros(8, dtype=np.float32)]
        state.raw_losses = {"moons": [0.2]}
        state.benchmark_results = []
        state.total_evaluations += 1
        return state

    monkeypatch.setattr(coordinator_mod, "evaluate", fake_evaluate)
    monkeypatch.setattr(coordinator_mod, "score", lambda state, config: state)

    coordinator_mod.run_evolution(
        cfg,
        benchmark_spec=SimpleNamespace(name="moons", task="classification", input_dim=2, num_classes=2),
        run_dir=str(run_dir),
        resume=True,
    )

    with RunStore(run_dir / "metrics.duckdb") as store:
        budget_meta = store.load_budget_metadata("current")

    assert budget_meta["primordia_seeding"] is None


def test_weight_cache_partial_lookup_and_fifo_eviction():
    genome = _seed_genome(3)
    variant = copy.deepcopy(genome)
    variant.layers[1] = variant.layers[1].model_copy(update={"width": variant.layers[1].width + 8})
    other = _seed_genome(11)

    cache = WeightCache(max_size=1)
    cache.store(genome, {"w": np.array([1.0], dtype=np.float32)}, namespace="demo")

    exact = cache.lookup(genome, namespace="demo")
    partial = cache.lookup_partial(variant, namespace="demo")

    assert exact is not None and float(exact["w"][0]) == 1.0
    assert partial is not None and float(partial["w"][0]) == 1.0

    cache.store(other, {"w": np.array([2.0], dtype=np.float32)}, namespace="demo")

    assert len(cache) == 1
    assert cache.lookup(genome, namespace="demo") is None


def test_reproduce_keeps_protected_survivor_and_resets_metrics():
    population = [_seed_genome(1), _seed_genome(2), _seed_genome(3)]
    for idx, genome in enumerate(population):
        genome.fitness = 0.1 + idx * 0.1
        genome.param_count = 32 + idx
        genome.model_bytes = 64 + idx

    state = GenerationState(
        generation=0,
        population=population,
        fitnesses=[0.1, 0.2, 0.3],
        model_bytes=[64, 65, 66],
    )
    cfg = RunConfig.model_validate(
        {
            "evolution": {
                "population_size": 3,
                "elite_count": 1,
                "crossover_ratio": 0.0,
                "mutation_rates": {
                    "width": 0.0,
                    "activation": 0.0,
                    "add_layer": 0.0,
                    "remove_layer": 0.0,
                    "add_connection": 0.0,
                    "remove_connection": 0.0,
                    "add_residual": 0.0,
                    "weight_bits": 0.0,
                    "activation_bits": 0.0,
                    "sparsity": 0.0,
                    "operator_type": 0.0,
                },
            }
        }
    )

    next_state, pending = reproduce(
        state,
        cfg,
        InnovationCounter(),
        MutationScheduler(),
        random.Random(5),
        protected_indices={2},
    )

    survivor_structures = [len(genome.enabled_connections) for genome in next_state.population[:2]]

    assert len(next_state.population) == 3
    assert survivor_structures[0] == len(population[0].enabled_connections)
    assert survivor_structures[1] == len(population[2].enabled_connections)
    assert next_state.fitnesses == []
    assert next_state.model_bytes == []
    assert all(outcome.genome_idx >= 2 for outcome in pending)


def test_family_percentile_aggregation_balances_families():
    raw_losses = {
        "tab_a": [0.1, 0.9],
        "tab_b": [0.8, 0.2],
        "lm_a": [0.9, 0.1],
    }
    benchmark_families = {
        "tab_a": "tabular",
        "tab_b": "tabular",
        "lm_a": "language_modeling",
    }

    baseline_scores = _aggregate_pool_fitness(
        raw_losses=raw_losses,
        benchmark_families=benchmark_families,
        benchmark_timings=[
            {"benchmark_name": "tab_a", "evaluation_seconds": 0.1},
            {"benchmark_name": "tab_b", "evaluation_seconds": 1.0},
            {"benchmark_name": "lm_a", "evaluation_seconds": 0.5},
        ],
        aggregation="family_percentile",
        active_family=None,
        family_focus_weight=2.0,
        benchmark_cost_penalty_alpha=0.0,
    )
    scores = _aggregate_pool_fitness(
        raw_losses=raw_losses,
        benchmark_families=benchmark_families,
        benchmark_timings=[
            {"benchmark_name": "tab_a", "evaluation_seconds": 0.1},
            {"benchmark_name": "tab_b", "evaluation_seconds": 1.0},
            {"benchmark_name": "lm_a", "evaluation_seconds": 0.5},
        ],
        aggregation="family_percentile",
        active_family=None,
        family_focus_weight=2.0,
        benchmark_cost_penalty_alpha=0.5,
    )

    assert (scores[0] - scores[1]) < (baseline_scores[0] - baseline_scores[1])


def test_family_transfer_uses_family_namespace_cache():
    genome = _seed_genome(17)
    cache = WeightCache()
    cached_weights = {"w": np.ones(1, dtype=np.float32)}
    cache.store(genome, cached_weights, namespace="family::tabular")

    plan = _make_evaluation_plan(
        genome=genome,
            config=RunConfig.model_validate(
                {
                    "benchmark_pool": {"benchmarks": ["demo"], "sample_k": 1, "family_transfer": True},
                    "training": {
                        "epochs": 10,
                        "finetune_epoch_ratio": 0.5,
                        "multi_fidelity": False,
                    },
                }
            ),
        input_dim=2,
        num_classes=2,
        task="classification",
        cache=cache,
        cache_namespace="demo_benchmark",
        family_namespace="family::tabular",
        multi_fidelity_schedule=None,
        generation=0,
        evaluation_memo=None,
    )

    assert plan["reused"] is False
    assert plan["weight_snapshot"] is not None
    assert int(plan["epochs"]) == 5
