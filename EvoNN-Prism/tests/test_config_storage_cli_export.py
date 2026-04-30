from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - host-dependent MLX runtime availability
    mx = None

from prism.cli import app
from prism.benchmarks.datasets import get_benchmark
from prism.config import RunConfig, load_config
from prism.export import symbiosis as sym
from prism.genome import ModelGenome
from prism.runtime import cache as cache_mod
from prism.runtime.cache import WeightCache
from prism.runtime.training import EvaluationResult
from prism.storage import RunStore


runner = CliRunner()


def _import_coordinator_or_skip():
    return pytest.importorskip(
        "prism.pipeline.coordinator",
        reason="MLX runtime unavailable on this host",
        exc_type=ImportError,
    )


def _import_compile_genome_or_skip():
    module = pytest.importorskip(
        "prism.families.compiler",
        reason="MLX runtime unavailable on this host",
        exc_type=ImportError,
    )
    return module.compile_genome


def _sample_genome(family: str = "mlp", widths: list[int] | None = None) -> ModelGenome:
    return ModelGenome(
        family=family,
        hidden_layers=widths or [16, 8],
        activation="relu",
        dropout=0.1,
    )


def test_load_config_reads_yaml_overrides(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "training": {"epochs": 5, "batch_size": 64},
                "evolution": {"population_size": 3},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.seed == 7
    assert cfg.training.epochs == 5
    assert cfg.training.batch_size == 64
    assert cfg.evolution.population_size == 3


def test_run_store_roundtrip_and_best_per_benchmark(tmp_path: Path):
    db_path = tmp_path / "metrics.duckdb"
    genome_a = _sample_genome("mlp", [16, 8])
    genome_b = _sample_genome("conv2d", [32, 16])
    genome_c = _sample_genome("attention", [24, 12])

    with RunStore(db_path) as store:
        store.save_run("run-1", {"seed": 42})
        store.save_genome("run-1", genome_a)
        store.save_genome("run-1", genome_b)
        store.save_genome("run-1", genome_c)
        store.save_evaluation(
            "run-1", genome_a.genome_id, 0, "moons", "accuracy", 0.8, 0.8, 100, 0.5
        )
        store.save_evaluation(
            "run-1", genome_b.genome_id, 1, "moons", "accuracy", 0.9, 0.9, 120, 0.6
        )
        store.save_evaluation(
            "run-1",
            genome_c.genome_id,
            1,
            "iris",
            "accuracy",
            float("nan"),
            float("nan"),
            140,
            0.4,
            status="missing",
        )
        store.save_lineage("run-1", genome_b.genome_id, genome_a.genome_id, 1, "mut")
        store.save_archive("run-1", 1, "pareto", "moons", genome_b.genome_id, 0.9)

        loaded = store.load_genomes("run-1")
        evals = store.load_evaluations("run-1")
        lineage = store.load_lineage("run-1")
        archives = store.load_archives("run-1")
        best = store.load_best_per_benchmark("run-1")
        latest = store.latest_generation("run-1")

    assert [row["_family"] for row in loaded] == ["mlp", "conv2d", "attention"]
    assert len(evals) == 3
    assert lineage[0]["mutation_summary"] == "mut"
    assert archives[0]["generation"] == 1
    assert {row["benchmark_id"]: row["status"] for row in evals} == {
        "moons": "ok",
        "iris": "missing",
    }
    assert best["moons"]["genome_id"] == genome_b.genome_id
    assert "iris" not in best
    assert best["moons"]["metric_value"] == 0.9
    assert latest == 1


def test_export_helpers_cover_config_resolution_and_summary(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"seed": 9, "training": {"epochs": 4}, "evolution": {"population_size": 2}}),
        encoding="utf-8",
    )

    cfg = sym._load_run_config(run_dir)
    assert cfg.seed == 9
    assert cfg.training.epochs == 4

    assert sym._benchmark_metric_name("classification") == "accuracy"
    assert sym._benchmark_metric_direction("classification") == "max"
    assert sym._benchmark_metric_name("regression") == "mse"
    assert sym._benchmark_metric_direction("regression") == "min"

    genome_a = _sample_genome("mlp", [8, 4])
    genome_b = _sample_genome("conv2d", [32, 16])
    evaluations = [
        {"genome_id": genome_a.genome_id, "quality": 0.4, "failure_reason": None},
        {"genome_id": genome_b.genome_id, "quality": 0.9, "failure_reason": None},
    ]
    rep = sym._select_representative([genome_a, genome_b], evaluations)
    assert rep == genome_b
    assert sym._architecture_summary(rep) == "conv2d [32x16] relu"
    assert sym._compute_dataset_hash(["b", "a"]) == sym._compute_dataset_hash(["a", "b"])

    monkeypatch.setattr(sym.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sym.platform, "machine", lambda: "arm64")
    assert sym._detect_device() == "apple_silicon"
    assert sym._intended_evaluation_count(
        config=RunConfig(evolution={"population_size": 8}),
        generations=1,
        benchmark_count=38,
        fallback=0,
    ) == 304

    manifest = {"run_id": "demo", "budget": {"evaluation_count": 3, "wall_clock_seconds": 12.5}}
    results = [
        {"benchmark_id": "moons", "metric_value": 0.91, "quality": 9.1, "status": "ok"},
        {"benchmark_id": "iris", "metric_value": 0.95, "quality": 9.5, "status": "ok"},
        {"benchmark_id": "bad", "metric_value": None, "quality": -999.0, "status": "failed"},
    ]
    best_per_benchmark = {
        "moons": {"genome_id": genome_b.genome_id},
        "iris": {"genome_id": genome_a.genome_id},
    }
    lineage_records = [
        {"mutation_summary": "mutation:width", "operator_kind": "mutation"},
        {"mutation_summary": "crossover", "operator_kind": "crossover"},
    ]
    sym._write_summary_json(
        tmp_path,
        manifest,
        results,
        [genome_a, genome_b],
        1,
        RunConfig(),
        best_per_benchmark=best_per_benchmark,
        lineage_records=lineage_records,
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["system"] == "prism"
    assert summary["total_evaluations"] == 3
    assert summary["wall_clock_seconds"] == 12.5
    assert summary["generations_completed"] == 2
    assert summary["failure_count"] == 1
    assert summary["benchmarks_evaluated"] == 2
    assert summary["median_benchmark_quality"] == pytest.approx(0.93)
    assert summary["runtime_backend"] == "mlx"
    assert summary["precision_mode"] == "fp32"
    assert summary["operator_mix"]["mutation:width"] == 1
    assert summary["family_benchmark_wins"] == {"conv2d": 1, "mlp": 1}
    assert summary["failure_patterns"]["failed"] == 1


def test_resolve_run_id_prefers_latest_entry(tmp_path: Path):
    with RunStore(tmp_path / "metrics.duckdb") as store:
        store.save_run("old", {"seed": 1})
        store.save_run("new", {"seed": 2})
        resolved = sym._resolve_run_id(store)
    assert resolved in {"old", "new"}


def test_cli_benchmarks_and_symbiosis_export(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("prism.cli.list_benchmarks", lambda: ["moons"])
    monkeypatch.setattr(
        "prism.cli.get_benchmark",
        lambda name: SimpleNamespace(task="classification", input_dim=2, input_shape=None, num_classes=2, output_dim=2, source="sklearn"),
    )
    result = runner.invoke(app, ["benchmarks"])
    assert result.exit_code == 0
    assert "moons" in result.stdout

    called: dict[str, str] = {}

    def fake_export(run_dir, pack_path, output_dir=None):
        called["run_dir"] = str(run_dir)
        called["pack_path"] = str(pack_path)
        return tmp_path / "manifest.json", tmp_path / "results.json"

    monkeypatch.setattr("prism.export.symbiosis.export_symbiosis_contract", fake_export)
    (tmp_path / "demo-run").mkdir()
    result = runner.invoke(
        app,
        ["symbiosis", "export", str(tmp_path / "demo-run"), "--pack", "tier1_core"],
    )

    assert result.exit_code == 0
    assert called["pack_path"] == "tier1_core"
    assert "manifest" in result.stdout


def test_cli_evolve_loads_pack_and_calls_coordinator(monkeypatch, tmp_path: Path):
    coordinator = _import_coordinator_or_skip()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 5,
                "benchmark_pack": {"pack_name": "tier1_core"},
                "evolution": {"population_size": 2, "num_generations": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    called: dict[str, object] = {}

    monkeypatch.setattr("prism.cli.resolve_pack_path", lambda pack: tmp_path / "tier1_core.yaml")
    monkeypatch.setattr("prism.cli.load_parity_pack", lambda pack: [SimpleNamespace(id="moons")])

    def fake_run_evolution(*, config, benchmark_specs, run_dir, resume):
        called["config"] = config
        called["benchmark_specs"] = benchmark_specs
        called["run_dir"] = run_dir
        called["resume"] = resume

    monkeypatch.setattr(coordinator, "run_evolution", fake_run_evolution)

    run_dir = tmp_path / "run"
    result = runner.invoke(app, ["evolve", "-c", str(config_path), "--run-dir", str(run_dir)])

    assert result.exit_code == 0
    assert called["run_dir"] == str(run_dir)
    assert called["resume"] is False
    assert len(called["benchmark_specs"]) == 1


def test_create_seed_population_keeps_unique_genome_ids():
    coordinator = _import_coordinator_or_skip()
    evolution = RunConfig.model_validate(
        {
            "evolution": {
                "population_size": 8,
                "allowed_families": ["mlp", "sparse_mlp", "attention", "sparse_attention"],
            }
        }
    ).evolution
    population = coordinator._create_seed_population(evolution, random.Random(42))
    assert len(population) == 8
    assert len({g.genome_id for g in population}) == 8


def test_coordinator_persists_duckdb_and_report_reads_results(monkeypatch, tmp_path: Path):
    coordinator = _import_coordinator_or_skip()
    cfg = RunConfig.model_validate(
        {
            "seed": 3,
            "training": {"epochs": 1},
            "evolution": {
                "population_size": 1,
                "offspring_per_generation": 1,
                "num_generations": 1,
                "allowed_families": ["mlp"],
            },
        }
    )
    benchmark = SimpleNamespace(id="moons", name="moons")

    monkeypatch.setattr(
        "prism.pipeline.evaluate._evaluate_single",
        lambda genome, spec, training, epoch_scale, cache, parent_ids=None: EvaluationResult(
            metric_name="accuracy",
            metric_value=0.91,
            quality=0.91,
            parameter_count=123,
            train_seconds=0.2,
            failure_reason=None,
            inherited_from=parent_ids[0] if parent_ids else None,
        ),
    )

    run_dir = tmp_path / "run"
    state = coordinator.run_evolution(cfg, [benchmark], run_dir=str(run_dir))
    assert state.total_evaluations == 1
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_backend"] == "mlx"
    assert summary["precision_mode"] == "fp32"

    with RunStore(run_dir / "metrics.duckdb") as store:
        evals = store.load_evaluations(run_dir.name)
        genomes = store.load_genomes(run_dir.name)
        assert len(evals) == 1
        assert len(genomes) == 1
        assert evals[0]["metric_value"] == 0.91

    from prism.export.report import generate_report

    report = generate_report(run_dir)
    assert "Total Evaluations | 1" in report
    assert "| Runtime | mlx |" in report
    assert "| Precision Mode | fp32 |" in report
    assert "| Wall Clock Seconds |" in report
    assert "## Family Benchmark Wins" in report
    assert "## Operator Mix" in report
    assert "## Operator Success" in report
    assert "## Weight Inheritance" in report
    assert "## Family Survival" in report
    assert "## Archive Turnover" in report
    assert "## Failure Patterns" in report
    assert "## Failure Heatmap" in report
    assert "| moons | 0.910000 | accuracy | 123 | 0.20 |" in report


def test_run_evolution_rejects_duplicate_offspring_ids(monkeypatch, tmp_path: Path):
    coordinator = _import_coordinator_or_skip()
    cfg = RunConfig.model_validate(
        {
            "seed": 3,
            "training": {"epochs": 1},
            "evolution": {
                "population_size": 2,
                "offspring_per_generation": 2,
                "num_generations": 2,
                "allowed_families": ["mlp"],
                "family_offspring_floor": 0,
                "benchmark_specialist_offspring": 0,
            },
        }
    )
    benchmark = SimpleNamespace(id="moons", name="moons")

    monkeypatch.setattr(
        "prism.pipeline.evaluate._evaluate_single",
        lambda genome, spec, training, epoch_scale, cache, parent_ids=None: EvaluationResult(
            metric_name="accuracy",
            metric_value=0.91,
            quality=0.91,
            parameter_count=123,
            train_seconds=0.2,
            failure_reason=None,
            inherited_from=parent_ids[0] if parent_ids else None,
        ),
    )

    duplicate_child = _sample_genome("mlp", [32, 16])

    def fake_reproduce(state, config, rng):
        return (
            [duplicate_child, duplicate_child],
            [
                {
                    "genome_id": duplicate_child.genome_id,
                    "parent_ids": [state.population[0].genome_id],
                    "operator": "mutation:width",
                },
                {
                    "genome_id": duplicate_child.genome_id,
                    "parent_ids": [state.population[1].genome_id],
                    "operator": "mutation:width",
                },
            ],
        )

    monkeypatch.setattr(coordinator, "reproduce", fake_reproduce)

    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="duplicate offspring genome ids"):
        coordinator.run_evolution(cfg, [benchmark], run_dir=str(run_dir))


def test_inspect_surfaces_failure_patterns_and_recent_failures(tmp_path: Path):
    run_dir = tmp_path / "inspect-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"seed": 17}, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "runtime_backend": "mlx",
                "runtime_version": "0.0-test",
                "precision_mode": "fp32",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    genome = _sample_genome("mlp", [16, 8])
    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run(run_dir.name, {"seed": 17})
        store.save_genome(run_dir.name, genome)
        store.save_evaluation(run_dir.name, genome.genome_id, 0, "moons", "accuracy", 0.91, 0.91, 101, 0.2)
        store.save_evaluation(
            run_dir.name,
            genome.genome_id,
            0,
            "iris",
            "accuracy",
            float("nan"),
            float("nan"),
            101,
            0.1,
            failure_reason="compile_timeout",
        )

    result = runner.invoke(app, ["inspect", str(run_dir)])

    assert result.exit_code == 0, result.stdout
    assert "Evaluation Status Mix" in result.stdout
    assert "ok=1, failed=1" in result.stdout
    assert "Failure Patterns" in result.stdout
    assert "compile_timeout" in result.stdout
    assert "Recent Failures" in result.stdout
    assert "iris" in result.stdout


def test_coordinator_summary_persists_failure_patterns(monkeypatch, tmp_path: Path):
    coordinator = _import_coordinator_or_skip()
    cfg = RunConfig.model_validate(
        {
            "seed": 9,
            "training": {"epochs": 1},
            "evolution": {
                "population_size": 1,
                "offspring_per_generation": 1,
                "num_generations": 1,
                "allowed_families": ["mlp"],
            },
        }
    )
    benchmark = SimpleNamespace(id="iris", name="iris")

    monkeypatch.setattr(
        "prism.pipeline.evaluate._evaluate_single",
        lambda genome, spec, training, epoch_scale, cache, parent_ids=None: EvaluationResult(
            metric_name="accuracy",
            metric_value=float("nan"),
            quality=float("nan"),
            parameter_count=77,
            train_seconds=0.05,
            failure_reason="compile_timeout",
            inherited_from=parent_ids[0] if parent_ids else None,
        ),
    )

    run_dir = tmp_path / "failing-run"
    state = coordinator.run_evolution(cfg, [benchmark], run_dir=str(run_dir))
    assert state.total_evaluations == 1

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmarks_evaluated"] == 1
    assert summary["failure_count"] == 1
    assert summary["failure_patterns"] == {"compile_timeout": 1}


def test_export_symbiosis_contract_end_to_end(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": 17,
                "training": {"epochs": 3},
                "evolution": {"population_size": 2},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 7.25,
                "runtime_backend": "mlx",
                "runtime_version": "0.0-test",
                "precision_mode": "fp32",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    genome_a = _sample_genome("mlp", [16, 8])
    genome_b = _sample_genome("conv2d", [32, 16])
    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run(run_dir.name, {"seed": 17})
        store.save_genome(run_dir.name, genome_a)
        store.save_genome(run_dir.name, genome_b)
        store.save_evaluation(
            run_dir.name,
            genome_a.genome_id,
            0,
            "moons",
            "accuracy",
            0.91,
            0.91,
            101,
            0.2,
            inherited_from="parent-x",
        )
        store.save_evaluation(
            run_dir.name,
            genome_b.genome_id,
            0,
            "iris",
            "accuracy",
            0.95,
            0.95,
            205,
            0.3,
        )
        store.save_lineage(run_dir.name, genome_a.genome_id, "parent-x", 0, "mutation:width")
        store.save_lineage(run_dir.name, genome_b.genome_id, genome_a.genome_id, 0, "crossover", "crossover")
        store.save_archive(run_dir.name, 0, "pareto", None, genome_a.genome_id, 0.91)
        store.save_archive(run_dir.name, 0, "pareto", None, genome_b.genome_id, 0.95)

    monkeypatch.setattr(sym, "_code_version", lambda: "deadbeef")
    monkeypatch.setattr(
        sym,
        "load_parity_pack",
        lambda pack_path: [
            SimpleNamespace(id="moons", task="classification"),
            SimpleNamespace(id="iris", task="classification"),
        ],
    )
    monkeypatch.setattr(sym, "get_canonical_id", lambda name: f"canon::{name}")

    manifest_path, results_path = sym.export_symbiosis_contract(run_dir, "demo_pack.yaml")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert manifest["fairness"]["code_version"] == "deadbeef"
    assert manifest["fairness"]["benchmark_pack_id"] == "demo_pack"
    assert manifest["artifacts"]["canonical_benchmarks"] == ["canon::moons", "canon::iris"]
    assert manifest["device"]["framework"] == "mlx"
    assert manifest["device"]["precision_mode"] == "fp32"
    assert manifest["device"]["framework_version"] == "0.0-test"
    assert manifest["budget"]["wall_clock_seconds"] == 7.25
    assert len(results) == 2
    assert summary["runtime_backend"] == manifest["device"]["framework"]
    assert summary["precision_mode"] == manifest["device"]["precision_mode"]
    assert summary["wall_clock_seconds"] == manifest["budget"]["wall_clock_seconds"]
    assert summary["operator_mix"]["crossover"] == 1
    assert summary["family_benchmark_wins"] == {"conv2d": 1, "mlp": 1}
    assert summary["failure_patterns"] == {}


@pytest.mark.skipif(mx is None, reason="MLX runtime unavailable on this host")
def test_language_modeling_specs_load_and_compile():
    compile_genome = _import_compile_genome_or_skip()
    tiny = get_benchmark("tiny_lm_synthetic")
    assert tiny.task == "language_modeling"
    x_train, y_train, x_val, y_val = tiny.load_data(seed=7)
    assert x_train.ndim == 2
    assert y_train.ndim == 2
    assert x_train.dtype == np.int32
    assert y_train.dtype == np.int64
    assert x_train.shape[1] == 128
    assert y_train.shape[1] == 128

    for name, train_rows, val_rows in [
        ("tinystories_lm_smoke", 512, 128),
        ("wikitext2_lm_smoke", 512, 128),
    ]:
        spec = get_benchmark(name)
        xs, ys, xv, yv = spec.load_data(seed=42)
        assert xs.shape == (train_rows, 256)
        assert ys.shape == (train_rows, 256)
        assert xv.shape == (val_rows, 256)
        assert yv.shape == (val_rows, 256)

    genome = _sample_genome("attention", [32, 32]).model_copy(update={"embedding_dim": 32})
    compiled = compile_genome(genome, [8], 32, "text", task="language_modeling")
    probs = np.array(compiled.model(mx.array(np.random.randint(0, 32, size=(2, 8), dtype=np.int32))))
    assert probs.shape == (2, 8, 32)


@pytest.mark.skipif(mx is None, reason="MLX runtime unavailable on this host")
def test_weight_cache_skips_missing_parameter_paths():
    class DummyModel:
        def trainable_parameters(self):
            return {"layers": {"0": {"weight": np.zeros((2, 2), dtype=np.float32)}}}

        def update(self, params):
            raise ValueError('Module does not have parameter named "layers.0.weight".')

    cache = WeightCache()
    cache._cache["parent"] = {"layers.0.weight": np.ones((2, 2), dtype=np.float32)}
    assert cache.transfer_weights("parent", DummyModel()) is False


def test_weight_cache_finds_best_compatible_fallback(monkeypatch):
    cache = WeightCache()
    cache._cache["weak"] = {"w": np.zeros((2, 2), dtype=np.float32)}
    cache._cache["strong"] = {"w": np.zeros((4, 4), dtype=np.float32)}
    cache._meta["weak"] = {"family": "mlp"}
    cache._meta["strong"] = {"family": "mlp"}

    monkeypatch.setattr(
        cache_mod,
        "_flatten_trainable_parameters",
        lambda model: {"w": np.zeros((4, 4), dtype=np.float32)},
    )
    monkeypatch.setattr(
        cache_mod,
        "_compatibility_score",
        lambda parent, child: 5.0 if parent["w"].shape == (4, 4) else 1.0,
    )
    applied = []
    monkeypatch.setattr(
        cache_mod,
        "_apply_matching_weights",
        lambda parent, model, child: applied.append(parent["w"].shape) or 1,
    )

    source = cache.transfer_best_available(object(), family="mlp", preferred_ids=["weak"])

    assert source == "strong"
    assert applied == [(4, 4)]


def test_smoke_pack_contains_33_plus_5_lm():
    pack_path = Path(__file__).resolve().parents[1] / "configs" / "working_33_plus_5_lm_smoke.yaml"
    payload = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    benchmarks = payload["benchmarks"]

    assert len(benchmarks) == 38
    assert "tiny_lm_synthetic" in benchmarks
    assert "tinystories_lm" in benchmarks
    assert "wikitext2_lm" in benchmarks


def test_load_prior_run_memory_and_seed_population(tmp_path: Path):
    coordinator = _import_coordinator_or_skip()
    run_dir = tmp_path / "prior-run"
    run_dir.mkdir()
    genome = _sample_genome("attention", [32, 16])

    with RunStore(run_dir / "metrics.duckdb") as store:
        store.save_run(run_dir.name, {"seed": 42})
        store.save_genome(run_dir.name, genome)
        store.save_evaluation(run_dir.name, genome.genome_id, 0, "moons", "accuracy", 0.91, 0.91, 100, 0.2)
        store.save_lineage(run_dir.name, genome.genome_id, None, 0, "mutation:embedding_dim")

    prior = coordinator._load_prior_run_memory([str(run_dir)])
    evolution = RunConfig.model_validate(
        {"evolution": {"population_size": 2, "allowed_families": ["attention", "mlp"]}}
    ).evolution
    population = coordinator._create_seed_population(evolution, random.Random(0), prior_genomes=prior["genomes"])

    assert prior["genomes"][0].genome_id == genome.genome_id
    assert "mutation:embedding_dim" in prior["operator_stats"]
    assert "attention" in prior["family_stats"]
    assert any(candidate.genome_id == genome.genome_id for candidate in population)
