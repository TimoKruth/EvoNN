from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import topograph.pipeline.evaluate as evaluate_mod
from topograph.cli import app
from topograph.benchmarks.parity import resolve_benchmark_pool_names
from topograph.config import RunConfig, load_config
from topograph.export import symbiosis as sym
from topograph.export import report as report_mod
from topograph.genome.codec import genome_to_dict
from topograph.genome.genome import Genome, InnovationCounter
from topograph.pipeline import coordinator as coordinator_mod
from topograph.pipeline.evaluate import BenchmarkDataCache, EvaluationMemo
from topograph.pipeline.evaluate import GenerationState
from topograph.storage import RunStore, _is_better


runner = CliRunner()


def _genome_dict(fitness=0.4, param_count=64, model_bytes=128):
    return {
        "layers": [],
        "connections": [],
        "fitness": fitness,
        "param_count": param_count,
        "model_bytes": model_bytes,
    }


def test_load_config_reads_yaml_and_validates_rotation_interval(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 11,
                "benchmark": "iris",
                "training": {"epochs": 7},
                "benchmark_pool": {"benchmarks": ["moons"], "rotation_interval": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)
    assert cfg.seed == 11
    assert cfg.benchmark == "iris"
    assert cfg.training.epochs == 7
    assert cfg.benchmark_pool.rotation_interval == 2

    with pytest.raises(ValueError):
        RunConfig.model_validate({"benchmark_pool": {"benchmarks": ["moons"], "rotation_interval": 0}})
    with pytest.raises(ValueError):
        RunConfig.model_validate({"training": {"parallel_cpu_fraction_limit": 0.0}})
    with pytest.raises(ValueError):
        RunConfig.model_validate({"benchmark_pool": {"benchmarks": [], "suite": None}})


def test_run_store_roundtrip_and_benchmark_best_selection(tmp_path: Path):
    db_path = tmp_path / "metrics.duckdb"
    with RunStore(db_path) as store:
        run_id = store.save_run(None, {"seed": 42})
        store.save_genomes(run_id, 0, [_genome_dict(fitness=0.7), _genome_dict(fitness=0.3)])
        store.save_innovation_counter(run_id, 12)
        store.save_budget_metadata(run_id, {"evaluation_count": 5})
        store.save_benchmark_results(
            run_id,
            0,
            [
                {
                    "benchmark_name": "moons",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": 0.81,
                    "quality": 0.81,
                    "parameter_count": 100,
                    "train_seconds": 0.5,
                    "architecture_summary": "3L/4C",
                    "genome_id": "g0",
                    "genome_idx": 0,
                    "status": "ok",
                    "failure_reason": None,
                },
                {
                    "benchmark_name": "moons",
                    "metric_name": "accuracy",
                    "metric_direction": "max",
                    "metric_value": 0.9,
                    "quality": 0.9,
                    "parameter_count": 120,
                    "train_seconds": 0.6,
                    "architecture_summary": "4L/5C",
                    "genome_id": "g1",
                    "genome_idx": 1,
                    "status": "ok",
                    "failure_reason": None,
                },
            ],
        )
        store.save_benchmark_timings(
            run_id,
            0,
            [
                {
                    "benchmark_order": 0,
                    "benchmark_name": "moons",
                    "task": "classification",
                    "data_load_seconds": 0.1,
                    "evaluation_seconds": 0.4,
                    "total_seconds": 0.5,
                    "trained_count": 2,
                    "reused_count": 1,
                    "failed_count": 0,
                    "requested_worker_count": 4,
                    "resolved_worker_count": 2,
                }
            ],
        )

        loaded_run = store.load_run(run_id)
        genomes = store.load_genomes(run_id, 0)
        latest = store.load_latest_generation(run_id)
        counter = store.load_innovation_counter(run_id)
        budget = store.load_budget_metadata(run_id)
        best = store.load_best_benchmark_results(run_id)
        timings = store.load_benchmark_timings(run_id)

    assert loaded_run["seed"] == 42
    assert len(genomes) == 2
    assert latest == 0
    assert counter == 12
    assert budget["evaluation_count"] == 5
    assert best[0]["metric_value"] == 0.9
    assert best[0]["genome_id"] == "g1"
    assert timings[0]["resolved_worker_count"] == 2


def test_resolve_benchmark_pool_names_supports_suite_and_deduplicates():
    cfg = RunConfig.model_validate(
        {
            "benchmark_pool": {
                "suite": "smoke",
                "benchmarks": ["wine", "iris", "custom_bench"],
                "sample_k": 2,
            }
        }
    )

    names = resolve_benchmark_pool_names(cfg.benchmark_pool)

    assert names[:3] == ["iris", "moons", "wine"]
    assert names[-1] == "custom_bench"
    assert names.count("wine") == 1


def test_is_better_handles_min_and_max_metrics():
    assert _is_better(
        {"status": "ok", "metric_direction": "max", "metric_value": 0.9},
        {"status": "ok", "metric_direction": "max", "metric_value": 0.8},
    )
    assert _is_better(
        {"status": "ok", "metric_direction": "min", "metric_value": 0.2},
        {"status": "ok", "metric_direction": "min", "metric_value": 0.3},
    )


def test_export_helpers_cover_budget_search_artifacts_and_summary(tmp_path: Path, monkeypatch):
    cfg = RunConfig(benchmark_pool={"suite": "smoke", "benchmarks": ["moons", "iris"], "sample_k": 2})
    budget = sym._budget_manifest(cfg, {"wall_clock_seconds": 1.5}, latest_gen=2, population_size=4)
    assert budget["evaluation_count"] == 60
    assert budget["population_size"] == cfg.evolution.population_size
    assert budget["budget_policy_name"] == "prototype_equal_budget"

    telemetry_none = sym._search_telemetry(RunConfig(training={"multi_fidelity": False}), {})
    assert telemetry_none is None

    cfg2 = RunConfig(
        novelty_weight=0.2,
        map_elites=True,
        training={"multi_fidelity": True, "multi_fidelity_schedule": [0.5, 1.0]},
    )
    telemetry = sym._search_telemetry(
        cfg2,
        {"map_elites_occupied_niches": 3, "map_elites_total_niches": 6},
    )
    assert telemetry["qd_enabled"] is True
    assert telemetry["map_elites_fill_ratio"] == 0.5

    rep = SimpleNamespace(
        enabled_layers=[],
        enabled_connections=[],
        experts=[],
        param_count=64,
        fitness=0.3,
    )
    monkeypatch.setattr(sym, "_genome_summary", lambda genome: {"status": "ok"})
    artifacts = sym._build_artifacts_section(rep, ["moons"], cfg, "tier1_core")
    assert artifacts["pack_name"] == "tier1_core"
    assert artifacts["canonical_benchmarks"]

    monkeypatch.setattr(sym.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sym.platform, "machine", lambda: "arm64")
    assert sym._detect_device() == "apple_silicon"

    manifest = {"run_id": "demo", "budget": {"evaluation_count": 5, "wall_clock_seconds": 2.0}}
    results = [
        {"benchmark_id": "moons", "metric_value": 0.91, "status": "ok"},
        {"benchmark_id": "iris", "metric_value": 0.87, "status": "ok"},
        {"benchmark_id": "bad", "metric_value": None, "status": "failed"},
    ]
    pop = [SimpleNamespace(param_count=64, enabled_layers=[]), SimpleNamespace(param_count=128, enabled_layers=[])]
    sym._write_summary_json(tmp_path, manifest, results, pop, latest_gen=1, config=cfg)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["system"] == "topograph"
    assert summary["failure_count"] == 1
    assert summary["benchmarks_evaluated"] == 2


def test_load_run_config_and_resolve_run_id_from_store(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with RunStore(run_dir / "metrics.duckdb") as store:
        run_id = store.save_run("current", {"seed": 55, "benchmark": "moons"})
        assert sym._resolve_run_id(store) == "current"
        cfg = sym._load_run_config(run_dir, store)
    assert run_id == "current"
    assert cfg.seed == 55
    assert cfg.benchmark == "moons"


def test_symbiosis_export_preserves_failed_benchmarks(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "name": "demo_pack",
                "benchmarks": [
                    {
                        "benchmark_id": "tinystories_lm",
                        "native_ids": {"topograph": "tinystories_lm"},
                        "task_kind": "language_modeling",
                        "metric_name": "perplexity",
                        "metric_direction": "min",
                    }
                ],
                "budget_policy": {"evaluation_count": 1, "epochs_per_candidate": 1},
                "seed_policy": {"mode": "shared", "required": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run("demo", {"seed": 42, "benchmark": "tinystories_lm"})
        store.save_genomes("demo", 0, [_genome_dict(fitness=0.3, param_count=64, model_bytes=128)])
        store.save_benchmark_results(
            "demo",
            0,
            [
                {
                    "benchmark_name": "tinystories_lm",
                    "metric_name": "perplexity",
                    "metric_direction": "min",
                    "metric_value": None,
                    "quality": None,
                    "parameter_count": None,
                    "train_seconds": None,
                    "architecture_summary": "failed",
                    "genome_id": "g0",
                    "genome_idx": 0,
                    "status": "failed",
                    "failure_reason": "lm backend blew up",
                }
            ],
        )

    monkeypatch.setattr(sym, "_write_report_md", lambda output_dir, source_run_dir: None)
    monkeypatch.setattr(sym, "_write_summary_json", lambda *args, **kwargs: None)

    _, results_path = sym.export_symbiosis_contract(run_dir=run_dir, pack_path=pack_path)
    rows = json.loads(results_path.read_text(encoding="utf-8"))

    assert rows[0]["status"] == "failed"
    assert rows[0]["metric_value"] is None
    assert rows[0]["failure_reason"] == "lm backend blew up"


def test_cli_benchmarks_and_symbiosis_export(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("topograph.cli.list_benchmarks", lambda: ["moons"])
    monkeypatch.setattr(
        "topograph.cli.get_benchmark",
        lambda name: SimpleNamespace(task="classification", input_dim=2, num_classes=2),
    )
    result = runner.invoke(app, ["benchmarks"])
    assert result.exit_code == 0
    assert "moons" in result.stdout

    called: dict[str, str] = {}

    def fake_export(run_dir, pack_path, output_dir=None):
        called["run_dir"] = str(run_dir)
        called["pack_path"] = str(pack_path)
        return tmp_path / "manifest.json", tmp_path / "results.json"

    monkeypatch.setattr("topograph.export.symbiosis.export_symbiosis_contract", fake_export)
    (tmp_path / "demo-run").mkdir()
    result = runner.invoke(
        app,
        ["symbiosis", "export", str(tmp_path / "demo-run"), "--pack", "tier1_core"],
    )
    assert result.exit_code == 0
    assert called["pack_path"] == "tier1_core"
    assert "manifest" in result.stdout


def test_evaluate_pool_namespaces_weight_cache_by_benchmark(monkeypatch):
    state = GenerationState(
        generation=0,
        population=[SimpleNamespace(param_count=0, enabled_layers=[], enabled_connections=[])],
    )
    cfg = RunConfig(benchmark_pool={"benchmarks": ["bench_a", "bench_b"], "sample_k": 2})
    specs = [
        SimpleNamespace(name="bench_a", task="classification", input_dim=2, num_classes=2),
        SimpleNamespace(name="bench_b", task="classification", input_dim=2, num_classes=2),
    ]

    namespaces: list[str] = []

    monkeypatch.setattr(
        evaluate_mod,
        "_prepare_benchmark_data",
        lambda *args, **kwargs: (
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        evaluate_mod,
        "_evaluate_single",
        lambda **kwargs: (
            namespaces.append(kwargs["cache_namespace"])
            or (
                SimpleNamespace(
                    native_fitness=0.25,
                    metric_value=0.75,
                    metric_name="accuracy",
                    metric_direction="max",
                    quality=0.75,
                    train_seconds=0.0,
                    failure_reason=None,
                ),
                16,
            )
        ),
    )
    monkeypatch.setattr(evaluate_mod, "estimate_model_bytes", lambda g: 16)
    monkeypatch.setattr(evaluate_mod, "compute_behavior", lambda g: np.zeros(8, dtype=np.float32))

    new_state = evaluate_mod.evaluate_pool(
        state,
        cfg,
        specs,
        cache=SimpleNamespace(),
    )

    assert namespaces == ["bench_a", "bench_b"]
    assert new_state.total_evaluations == 2


def test_run_store_state_snapshot_roundtrip(tmp_path: Path):
    db_path = tmp_path / "metrics.duckdb"
    snapshot = {"next_generation": 2, "completed": False, "fitness_history": [0.3, 0.2]}
    with RunStore(db_path) as store:
        store.save_run_state("current", snapshot)
        loaded = store.load_run_state("current")

    assert loaded == snapshot


def test_cli_evolve_uses_coordinator(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": "moons",
                "evolution": {"population_size": 1, "num_generations": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    called: dict[str, object] = {}

    def fake_run_evolution(*, config, benchmark_spec, run_dir, resume):
        called["benchmark_spec"] = benchmark_spec.name
        called["run_dir"] = run_dir
        called["resume"] = resume
        return GenerationState(generation=0, population=[])

    monkeypatch.setattr("topograph.pipeline.coordinator.run_evolution", fake_run_evolution)
    monkeypatch.setattr(
        "topograph.cli.get_benchmark",
        lambda name: SimpleNamespace(name=name, task="classification", input_dim=2, num_classes=2),
    )

    result = runner.invoke(app, ["evolve", "-c", str(config_path), "--run-dir", str(tmp_path / "run")])
    assert result.exit_code == 0
    assert called["benchmark_spec"] == "moons"
    assert called["resume"] is False


def test_run_evolution_resumes_from_saved_snapshot(tmp_path: Path, monkeypatch):
    cfg = RunConfig(
        benchmark="moons",
        evolution={"population_size": 1, "num_generations": 2, "elite_count": 1},
        benchmark_elite_archive=False,
    )
    genome = Genome.create_seed(InnovationCounter(), random.Random(7))
    genome.learning_rate = 0.0125
    genome.batch_size = 32

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run("current", cfg.model_dump(mode="json"))
        store.save_run_state(
            "current",
            {
                "next_generation": 1,
                "population": [genome_to_dict(genome)],
                "innovation_counter": 99,
                "fitness_history": [0.45],
                "scheduler": {"stats": {}},
                "pool_state": {
                    "current_sample": [],
                    "rotation_counter": 0,
                    "benchmark_best_fitness": {},
                },
                "pending_outcomes": [],
                "elapsed_seconds": 1.25,
                "total_evaluations": 3,
                "novelty_score_sum": 0.0,
                "novelty_score_count": 0,
                "novelty_score_max": 0.0,
                "map_elites_insertions": 0,
                "completed": False,
            },
        )

    observed: list[tuple[int, float | None, int | None]] = []

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
        observed.append(
            (state.generation, state.population[0].learning_rate, state.population[0].batch_size)
        )
        state.fitnesses = [0.2]
        state.model_bytes = [16]
        state.behaviors = [np.zeros(8, dtype=np.float32)]
        state.raw_losses = {"moons": [0.2]}
        state.benchmark_results = []
        state.total_evaluations += 1
        return state

    monkeypatch.setattr(coordinator_mod, "evaluate", fake_evaluate)
    monkeypatch.setattr(coordinator_mod, "score", lambda state, config: state)

    final_state = coordinator_mod.run_evolution(
        cfg,
        benchmark_spec=SimpleNamespace(name="moons", task="classification", input_dim=2, num_classes=2),
        run_dir=str(run_dir),
        resume=True,
    )

    assert observed == [(1, 0.0125, 32)]
    assert final_state.total_evaluations == 4


def test_benchmark_data_cache_reuses_loaded_arrays():
    cache = BenchmarkDataCache()
    calls = {"count": 0}

    spec = SimpleNamespace(
        name="demo",
        task="classification",
        load_data=lambda seed, validation_split: (
            calls.__setitem__("count", calls["count"] + 1)
            or np.ones((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
            np.ones((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
        ),
    )

    first = cache.get(spec, seed=42, validation_split=0.2)
    second = cache.get(spec, seed=42, validation_split=0.2)

    assert calls["count"] == 1
    assert first[0] is second[0]


def test_evaluation_memo_reuses_exact_result_without_training(monkeypatch):
    genome = Genome.create_seed(InnovationCounter(), random.Random(3), num_layers=4)
    genome.learning_rate = 0.01
    genome.batch_size = 16
    memo = EvaluationMemo()
    result = evaluate_mod.EvaluationResult(
        metric_name="accuracy",
        metric_direction="max",
        metric_value=0.9,
        quality=0.9,
        native_fitness=0.1,
        train_seconds=0.0,
    )
    memo.store(
        benchmark_name="demo",
        genome=genome,
        epochs=2,
        lr=0.01,
        batch_size=16,
        result=result,
        model_bytes=32,
    )

    monkeypatch.setattr(evaluate_mod, "compile_genome", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not compile")))

    reused, mb = evaluate_mod._evaluate_single(
        genome=genome,
        config=RunConfig(training={"epochs": 2, "multi_fidelity": False}),
        X_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros(4, dtype=np.int64),
        X_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros(2, dtype=np.int64),
        input_dim=2,
        num_classes=2,
        task="classification",
        cache=None,
        multi_fidelity_schedule=None,
        generation=0,
        cache_namespace="demo",
        evaluation_memo=memo,
    )

    assert reused.native_fitness == 0.1
    assert mb == 32
    assert genome._last_eval_reused is True


def test_seed_population_default_layer_count_range():
    counter = InnovationCounter()
    rng = random.Random(11)

    for _ in range(40):
        genome = Genome.create_seed(counter, rng)
        assert 4 <= len(genome.enabled_layers) <= 12


def test_report_helpers_summarize_timings_and_quality_extremes():
    timing_rows = [
        {
            "benchmark_name": "fast",
            "task": "classification",
            "data_load_seconds": 0.1,
            "evaluation_seconds": 0.2,
            "total_seconds": 0.3,
            "trained_count": 1,
            "reused_count": 3,
            "failed_count": 0,
            "resolved_worker_count": 2,
        },
        {
            "benchmark_name": "slow",
            "task": "classification",
            "data_load_seconds": 1.0,
            "evaluation_seconds": 3.0,
            "total_seconds": 4.0,
            "trained_count": 4,
            "reused_count": 0,
            "failed_count": 1,
            "resolved_worker_count": 1,
        },
    ]
    summary = report_mod._summarize_benchmark_timings(timing_rows)
    assert summary["fastest"][0]["benchmark_name"] == "fast"
    assert summary["slowest"][0]["benchmark_name"] == "slow"

    extremes = report_mod._benchmark_quality_extremes(
        [
            {"benchmark_name": "a", "status": "ok", "quality": 0.8, "metric_name": "accuracy", "metric_value": 0.8},
            {"benchmark_name": "b", "status": "ok", "quality": -3.0, "metric_name": "perplexity", "metric_value": 20.0},
            {"benchmark_name": "c", "status": "failed", "quality": None, "metric_name": "accuracy", "metric_value": None},
        ]
    )
    assert extremes["best"][0]["benchmark_name"] == "a"
    assert extremes["worst"][0]["benchmark_name"] == "b"
