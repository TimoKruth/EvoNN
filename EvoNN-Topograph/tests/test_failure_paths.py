from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from topograph.benchmarks import parity as parity_mod
from topograph.benchmarks.registry import DatasetRegistry
from topograph.benchmarks.spec import BenchmarkSpec
from topograph.config import EarlyStoppingConfig, RunConfig
from topograph.export import report as report_mod
from topograph.genome.codec import genome_to_dict
from topograph.genome.genome import Genome, InnovationCounter
from topograph.monitor import TerminalMonitor
from topograph.parallel import ParallelEvaluator
from topograph.pipeline import coordinator as coordinator_mod
from topograph.pipeline.archive import BenchmarkEliteArchive, MAPElitesArchive, NoveltyArchive
from topograph.pipeline.evaluate import GenerationState
from topograph.pipeline.schedule import MutationScheduler
from topograph.storage import RunStore


def _seed_genome(seed: int = 7, num_layers: int = 3) -> Genome:
    genome = Genome.create_seed(InnovationCounter(), random.Random(seed), num_layers=num_layers)
    genome.fitness = 0.25
    genome.param_count = 64
    genome.model_bytes = 128
    genome.learning_rate = 0.01
    genome.batch_size = 32
    return genome


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_benchmark_spec_from_csv_and_unknown_source(tmp_path: Path):
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text("x1,x2,label\n1,2,a\n3,4,b\n5,6,a\n7,8,b\n", encoding="utf-8")

    spec = BenchmarkSpec.from_csv(csv_path)
    x_train, y_train, x_val, y_val = spec.load_data(seed=3, validation_split=0.5)

    assert spec.input_dim == 2
    assert spec.num_classes == 2
    assert x_train.shape == (2, 2)
    assert y_train.dtype == np.int64
    assert y_val.dtype == np.int64

    bad = BenchmarkSpec(name="bad", task="classification", source="unknown")
    with pytest.raises(ValueError, match="Unknown source"):
        bad.load_data()


def test_dataset_registry_prefers_shared_catalog_and_skips_invalid_yaml(tmp_path: Path, monkeypatch):
    shared_root = tmp_path / "shared-benchmarks"
    catalog_dir = shared_root / "catalog"
    _write_yaml(
        catalog_dir / "toy_moons.yaml",
        {
            "name": "toy_moons",
            "source": "sklearn",
            "dataset": "make_moons",
            "task": "classification",
            "input_dim": 2,
            "num_classes": 2,
            "n_samples": 20,
            "noise": 0.05,
            "domain": "tabular",
            "tags": ["toy", "classification"],
        },
    )
    _write_yaml(catalog_dir / "broken.yaml", {"name": "broken"})
    monkeypatch.setenv("EVONN_SHARED_BENCHMARKS_DIR", str(shared_root))
    monkeypatch.delenv("TOPOGRAPH_CATALOG_DIR", raising=False)

    registry = DatasetRegistry()
    names = [meta.name for meta in registry.list(task="classification", tag="toy")]
    x_train, y_train, x_val, y_val = registry.load_data("toy_moons", seed=11, validation_split=0.25)

    assert names == ["toy_moons"]
    assert x_train.dtype == np.float32
    assert y_train.dtype == np.int64
    assert x_train.shape[1] == 2
    assert x_val.shape[0] > 0
    assert y_val.shape[0] > 0
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        registry.get("broken")


def test_shared_benchmark_source_errors_are_explicit(tmp_path: Path, monkeypatch):
    missing_root = tmp_path / "missing-shared"
    monkeypatch.setenv("EVONN_SHARED_BENCHMARKS_DIR", str(missing_root))
    monkeypatch.delenv("TOPOGRAPH_CATALOG_DIR", raising=False)
    monkeypatch.delenv("TOPOGRAPH_SUITES_DIR", raising=False)

    with pytest.raises(FileNotFoundError, match="Shared benchmark catalog not found"):
        DatasetRegistry()
    with pytest.raises(FileNotFoundError, match="Shared benchmark suites not found"):
        parity_mod.load_benchmark_suite_names("smoke")


def test_benchmark_env_overrides_use_explicit_paths(tmp_path: Path, monkeypatch):
    catalog_dir = tmp_path / "catalog"
    suites_dir = tmp_path / "suites" / "topograph"
    _write_yaml(
        catalog_dir / "toy_moons.yaml",
        {
            "name": "toy_moons",
            "source": "sklearn",
            "dataset": "make_moons",
            "task": "classification",
            "input_dim": 2,
            "num_classes": 2,
            "n_samples": 12,
        },
    )
    _write_yaml(
        suites_dir / "toy.yaml",
        {"benchmarks": ["toy_moons"]},
    )
    monkeypatch.setenv("TOPOGRAPH_CATALOG_DIR", str(catalog_dir))
    monkeypatch.setenv("TOPOGRAPH_SUITES_DIR", str(tmp_path / "suites"))

    registry = DatasetRegistry()
    names = [meta.name for meta in registry.list()]
    suite_names = parity_mod.load_benchmark_suite_names("toy")

    assert names == ["toy_moons"]
    assert suite_names == ["toy_moons"]


def test_generate_report_covers_empty_and_populated_runs(tmp_path: Path):
    empty_run = tmp_path / "empty-run"
    empty_run.mkdir()
    with RunStore(empty_run / "metrics.duckdb") as store:
        store.save_run("current", {"seed": 1})
    empty_report = report_mod.generate_report(empty_run)
    assert "No generations found" in empty_report

    run_dir = tmp_path / "full-run"
    run_dir.mkdir()
    genome_a = _seed_genome(seed=1)
    genome_b = _seed_genome(seed=2)
    genome_b.fitness = 0.15
    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run("current", {"seed": 42, "benchmark": "moons"})
        store.save_genomes(
            "current",
            0,
            [genome_to_dict(genome_a), genome_to_dict(genome_b)],
        )
        store.save_budget_metadata(
            "current",
            {
                "runtime_backend": "mlx",
                "runtime_version": "0.17.1",
                "precision_mode": "mixed",
                "wall_clock_seconds": 3.2,
                "evaluation_count": 8,
                "evals_per_second": 2.5,
                "seconds_per_eval": 0.4,
                "cache_reuse_rate": 0.25,
                "cache_reused_count": 2,
                "cache_trained_count": 6,
                "data_cache_hits": 3,
                "data_cache_misses": 1,
                "requested_parallel_workers": 4,
                "resolved_parallel_workers_max": 2,
                "worker_clamp_reason_counts": {"memory": 1},
            },
        )
        store.save_benchmark_results(
            "current",
            0,
            [
                {
                    "benchmark_name": "moons",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": 0.91,
                    "quality": 0.91,
                    "parameter_count": 64,
                    "train_seconds": 0.3,
                    "architecture_summary": "3L/4C",
                    "genome_id": "g0",
                    "genome_idx": 0,
                    "status": "ok",
                    "failure_reason": None,
                }
            ],
        )
        store.save_benchmark_timings(
            "current",
            0,
            [
                {
                    "benchmark_order": 0,
                    "benchmark_name": "moons",
                    "task": "classification",
                    "data_load_seconds": 0.1,
                    "evaluation_seconds": 0.5,
                    "total_seconds": 0.6,
                    "trained_count": 2,
                    "reused_count": 1,
                    "failed_count": 0,
                    "requested_worker_count": 4,
                    "resolved_worker_count": 2,
                    "data_cache_hits": 3,
                    "data_cache_misses": 1,
                    "worker_clamp_reason": "memory",
                }
            ],
        )
        store.save_benchmark_results(
            "current",
            1,
            [
                {
                    "benchmark_name": "iris",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": 0.62,
                    "quality": 0.62,
                    "parameter_count": 48,
                    "train_seconds": 0.2,
                    "architecture_summary": "2L/2C",
                    "genome_id": "g2",
                    "genome_idx": 0,
                    "status": "ok",
                    "failure_reason": None,
                }
            ],
        )
        store.save_benchmark_timings(
            "current",
            1,
            [
                {
                    "benchmark_order": 0,
                    "benchmark_name": "iris",
                    "task": "classification",
                    "data_load_seconds": 0.05,
                    "evaluation_seconds": 0.25,
                    "total_seconds": 0.3,
                    "trained_count": 1,
                    "reused_count": 0,
                    "failed_count": 0,
                    "requested_worker_count": 4,
                    "resolved_worker_count": 1,
                    "data_cache_hits": 0,
                    "data_cache_misses": 1,
                    "worker_clamp_reason": "tasks",
                }
            ],
        )
        store.save_run_state(
            "current",
            {
                "benchmark_elite_archive": {
                    "elites": {
                        "moons": {
                            "benchmark_name": "moons",
                            "benchmark_family": "tabular",
                            "genome_idx": 1,
                            "fitness": 0.15,
                            "generation": 0,
                            "genome": genome_to_dict(genome_b),
                            "param_count": genome_b.param_count,
                            "model_bytes": genome_b.model_bytes,
                            "behavior": [2, 2, 0, 0, 3, 4, 0.5, 0],
                            "architecture_summary": "3L/4C",
                        }
                    }
                }
            },
        )

    report = report_mod.generate_report(run_dir)
    assert "## Benchmark Timing" in report
    assert "## Benchmark Fitness Extremes" in report
    assert "Parallel Workers" in report
    assert "Cache Reuse Rate" in report
    assert "Data Cache" in report
    assert "Precision Mode" in report
    assert "mixed" in report
    assert "## Sampled Benchmark Order" in report
    assert "## Worst Benchmark Trend" in report
    assert "## Topology Atlas" in report


def test_coordinator_helper_paths_cover_sampling_progress_and_budget(tmp_path: Path):
    cfg = RunConfig.model_validate(
        {
            "benchmark_pool": {
                "benchmarks": ["a", "b", "c"],
                "sample_k": 2,
                "rotation_interval": 2,
                "undercovered_benchmark_bias": 1.0,
            },
            "training": {
                "parallel_cpu_fraction_limit": 0.25,
                "parallel_memory_fraction_limit": 0.4,
                "parallel_reserved_system_memory_bytes": 123,
                "parallel_worker_thread_limit": 2,
            },
            "quantization_schedule": [
                {
                    "generations": [0, None],
                    "allowed_weight_bits": [4, 8],
                    "allowed_activation_bits": [8, 16],
                }
            ],
        }
    )
    specs = [
        SimpleNamespace(name="a", task="classification", source="sklearn"),
        SimpleNamespace(name="b", task="classification", source="image"),
        SimpleNamespace(name="c", task="language_modeling", source="lm_cache"),
    ]
    pool_state = {
        "current_sample": [],
        "rotation_counter": 0,
        "benchmark_best_fitness": {"a": 0.9, "b": 0.2, "c": 0.1},
        "benchmark_cost_seconds": {"a": [0.1], "b": [1.0], "c": [2.0]},
        "family_stage_history": [],
    }

    first = coordinator_mod._sample_benchmark_specs(
        cfg, specs, random.Random(1), pool_state, generation=0
    )
    second = coordinator_mod._sample_benchmark_specs(
        cfg, specs, random.Random(9), pool_state, generation=1
    )
    third = coordinator_mod._sample_benchmark_specs(
        cfg, specs, random.Random(11), pool_state, generation=2
    )

    assert "a" in {spec.name for spec in first}
    assert len(first) == 2
    assert [spec.name for spec in second] == [spec.name for spec in first]
    assert len(third) == 2
    assert any(spec.name == "b" for spec in third)

    coordinator_mod._update_pool_fitness_history(
        pool_state,
        {"a": [0.5, float("inf")], "b": [float("inf"), 0.3]},
    )
    coordinator_mod._update_pool_cost_history(
        pool_state,
        [
            {"benchmark_name": "a", "evaluation_seconds": 0.2},
            {"benchmark_name": "a", "evaluation_seconds": 0.3},
            {"benchmark_name": "b", "evaluation_seconds": 0.8},
        ],
    )
    assert pool_state["benchmark_best_fitness"]["a"] == 0.5
    assert pool_state["benchmark_best_fitness"]["b"] == 0.2
    assert pool_state["benchmark_cost_seconds"]["a"][-2:] == [0.2, 0.3]

    cfg_pop = RunConfig.model_validate(
        cfg.model_dump(mode="json") | {"evolution": {"population_size": 3}}
    )
    population = coordinator_mod._create_seed_population(
        cfg_pop,
        InnovationCounter(),
        random.Random(5),
    )
    assert len(population) == 3
    assert population[0].layers[0].operator.value == "residual"
    assert population[1].layers[0].operator.value == "attention_lite"
    assert all(genome.learning_rate is not None for genome in population)
    assert all(genome.batch_size is not None for genome in population)

    state = GenerationState(
        generation=0,
        population=population,
        benchmark_timings=[{"total_seconds": 1.2}],
        cache_reused=3,
        cache_trained=5,
        raw_losses={"a": [0.3], "b": [0.4]},
        total_evaluations=8,
    )
    elite_archive = BenchmarkEliteArchive()
    elite_archive.update(
        "a",
        0,
        0.3,
        0,
        benchmark_family="tabular",
        genome=population[0],
        behavior=np.zeros(8, dtype=np.float32),
        architecture_summary="3L/4C",
    )
    stats = coordinator_mod._generation_stats(first, elite_archive, state)
    assert "Benchmarks" in stats
    assert stats["Benchmark Elites"] == 1
    assert stats["Bench Time"] == "1.2s"
    assert stats["Cache Reuse"] == "3/8"

    assert coordinator_mod._should_stop_early([1.0, 0.95, 0.94], EarlyStoppingConfig(window=3, threshold=0.2))
    assert not coordinator_mod._should_stop_early([1.0, 0.8, 0.5], EarlyStoppingConfig(window=3, threshold=0.2))

    limits = coordinator_mod._parallel_runtime_limits(cfg)
    assert limits.cpu_fraction_limit == 0.25
    assert limits.memory_fraction_limit == 0.4
    assert limits.reserved_system_memory_bytes == 123
    assert limits.worker_thread_limit == 2

    calls: list[tuple[str, tuple]] = []

    class FakeMonitor:
        def on_benchmark_start(self, **kwargs):
            calls.append(("start", (kwargs["benchmark_name"], kwargs["benchmark_order"])))

        def on_benchmark_complete(self, **kwargs):
            calls.append(("complete", (kwargs["benchmark_name"], kwargs["resolved_worker_count"])))

    coordinator_mod._benchmark_progress(
        FakeMonitor(),
        0,
        "start",
        {"benchmark_name": "moons", "benchmark_order": 1, "benchmark_total": 3, "task": "classification"},
    )
    coordinator_mod._benchmark_progress(
        FakeMonitor(),
        0,
        "complete",
        {
            "benchmark_name": "moons",
            "benchmark_order": 1,
            "benchmark_total": 3,
            "total_seconds": 0.6,
            "data_load_seconds": 0.1,
            "evaluation_seconds": 0.5,
            "reused_count": 1,
            "trained_count": 2,
            "failed_count": 0,
            "resolved_worker_count": 2,
        },
    )
    assert calls == [("start", ("moons", 1)), ("complete", ("moons", 2))]

    with RunStore(tmp_path / "metrics.duckdb") as store:
        store.save_run("current", {"seed": 42})
        store.save_benchmark_timings(
            "current",
            0,
            [
                {
                    "benchmark_order": 0,
                    "benchmark_name": "a",
                    "task": "classification",
                    "data_load_seconds": 0.1,
                    "evaluation_seconds": 0.4,
                    "total_seconds": 0.5,
                    "trained_count": 2,
                    "reused_count": 1,
                    "failed_count": 0,
                    "requested_worker_count": 4,
                    "resolved_worker_count": 2,
                    "data_cache_hits": 1,
                    "data_cache_misses": 0,
                    "worker_clamp_reason": "memory",
                }
            ],
        )
        store.save_benchmark_results(
            "current",
            0,
            [
                {
                    "benchmark_name": "a",
                    "metric_name": "loss",
                    "metric_direction": "min",
                    "metric_value": 0.5,
                    "quality": 0.5,
                    "parameter_count": 12,
                    "train_seconds": 0.1,
                    "architecture_summary": "1L/1C",
                    "genome_id": "g0",
                    "genome_idx": 0,
                    "status": "ok",
                    "failure_reason": None,
                }
            ],
        )
        novelty = NoveltyArchive()
        novelty.add(np.zeros(8, dtype=np.float32))
        map_elites = MAPElitesArchive()
        map_elites.add(population[0], np.zeros(8, dtype=np.float32), 0.2)
        scheduler = MutationScheduler()
        scheduler.record_outcome("width", True)
        pool_state["family_stage_history"] = [
            {"generation": 0, "active_family": "tabular", "sampled_benchmarks": ["a"]}
        ]
        coordinator_mod._save_budget_metadata(
            store=store,
            run_id="current",
            state=state,
            config=cfg_pop,
            completed_generations=1,
            elapsed=4.0,
            novelty_archive=novelty,
            novelty_score_sum=0.9,
            novelty_score_count=3,
            novelty_score_max=0.5,
            map_elites_archive=map_elites,
            map_elites_insertions=2,
            benchmark_elite_archive=elite_archive,
            scheduler=scheduler,
            parallel_eval=ParallelEvaluator(max_workers=4),
            pool_state=pool_state,
        )
        budget = store.load_budget_metadata("current")

    assert budget["cache_reuse_rate"] == 0.333333
    assert budget["resolved_parallel_workers_max"] == 2
    assert budget["evals_per_second"] == 2.0
    assert budget["benchmark_elites"] == 1
    assert budget["data_cache_hits"] == 1
    assert budget["sampled_benchmark_order_by_generation"] == [{"generation": 0, "benchmarks": ["a"]}]
    assert budget["worker_clamp_reason_counts"] == {"memory": 1}
    assert budget["benchmark_elite_families"] == {"tabular": 1}
    assert budget["family_stage_history"] == pool_state["family_stage_history"]
