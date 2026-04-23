from __future__ import annotations

import math

from typer.testing import CliRunner

from prism.cli import app
from prism.export import report
from prism.export import symbiosis as sym


runner = CliRunner()


def test_runtime_metadata_defaults_to_unknown_when_summary_is_missing(tmp_path):
    run_dir = tmp_path / "runtime-metadata-missing"
    run_dir.mkdir()

    assert report._load_runtime_metadata(run_dir) == {
        "runtime_backend": "unknown",
        "runtime_version": "unknown",
        "precision_mode": "fp32",
        "wall_clock_seconds": None,
    }
    assert sym._load_runtime_metadata(run_dir) == {
        "runtime_backend": "unknown",
        "runtime_version": "unknown",
        "precision_mode": "fp32",
    }


def test_resolved_runtime_metadata_prefers_recorded_version_over_host_default(monkeypatch, tmp_path):
    run_dir = tmp_path / "runtime-metadata-recorded-version"
    run_dir.mkdir()

    monkeypatch.setattr(
        sym,
        "_load_runtime_metadata",
        lambda _run_dir: {
            "runtime_backend": "mlx",
            "runtime_version": "0.9.1-recorded",
            "precision_mode": "fp16",
        },
    )
    monkeypatch.setattr(sym, "_MLX_VERSION", "9.9.9-host")

    assert sym._resolved_runtime_metadata(run_dir) == {
        "runtime_backend": "mlx",
        "runtime_version": "0.9.1-recorded",
        "precision_mode": "fp16",
    }


def test_resolved_runtime_metadata_falls_back_to_host_version_when_recorded_unknown(monkeypatch, tmp_path):
    run_dir = tmp_path / "runtime-metadata-unknown-version"
    run_dir.mkdir()

    monkeypatch.setattr(
        sym,
        "_load_runtime_metadata",
        lambda _run_dir: {
            "runtime_backend": "unknown",
            "runtime_version": "unknown",
            "precision_mode": None,
        },
    )
    monkeypatch.setattr(sym, "_MLX_VERSION", "9.9.9-host")

    assert sym._resolved_runtime_metadata(run_dir) == {
        "runtime_backend": "mlx",
        "runtime_version": "9.9.9-host",
        "precision_mode": "fp32",
    }


def test_symbiosis_representative_ignores_non_ok_status_without_failure_reason():
    genome_ok = type("Genome", (), {"genome_id": "genome-ok", "family": "mlp"})()
    genome_missing = type("Genome", (), {"genome_id": "genome-missing", "family": "attention"})()
    evaluations = [
        {
            "genome_id": "genome-ok",
            "quality": 0.8,
            "failure_reason": None,
            "status": "ok",
        },
        {
            "genome_id": "genome-missing",
            "quality": 0.99,
            "failure_reason": None,
            "status": "missing",
        },
    ]

    representative = sym._select_representative([genome_ok, genome_missing], evaluations)

    assert representative is genome_ok


def test_report_failure_helpers_count_non_ok_status_without_failure_reason():
    evaluations = [
        {"benchmark_id": "moons", "status": "ok", "failure_reason": None},
        {"benchmark_id": "iris", "status": "missing", "failure_reason": None},
        {"benchmark_id": "iris", "status": "skipped", "failure_reason": None},
        {"benchmark_id": "digits", "status": "failed", "failure_reason": "compile_timeout:mlx"},
    ]

    assert report._compute_failure_patterns(evaluations) == {
        "missing": 1,
        "skipped": 1,
        "compile_timeout:mlx": 1,
    }
    assert report._compute_failure_heatmap(evaluations) == [
        {"benchmark": "digits", "failures": {"compile_timeout": 1}},
        {"benchmark": "iris", "failures": {"missing": 1, "skipped": 1}},
    ]


def test_report_success_helpers_ignore_non_ok_status_without_failure_reason():
    evaluations = [
        {
            "genome_id": "genome-ok",
            "generation": 0,
            "benchmark_id": "moons",
            "quality": 0.9,
            "train_seconds": 0.3,
            "parameter_count": 120,
            "failure_reason": None,
            "status": "ok",
            "inheritance_hit": True,
        },
        {
            "genome_id": "genome-missing",
            "generation": 0,
            "benchmark_id": "digits",
            "quality": 0.99,
            "train_seconds": 9.9,
            "parameter_count": 999,
            "failure_reason": None,
            "status": "missing",
            "inheritance_hit": False,
        },
    ]
    genomes = [
        type("Genome", (), {"genome_id": "genome-ok", "family": "mlp"})(),
        type("Genome", (), {"genome_id": "genome-missing", "family": "attention"})(),
    ]
    lineage = [
        {"genome_id": "genome-ok", "mutation_summary": "mutation:width"},
        {"genome_id": "genome-missing", "mutation_summary": "mutation:embedding_dim"},
    ]

    summary = report._compute_efficiency_summary(evaluations)
    family_rows = report._compute_family_efficiency(evaluations, genomes)
    operator_rows = report._compute_operator_efficiency(evaluations, genomes, lineage)
    operator_success_rows = report._compute_operator_success(evaluations, genomes, lineage)
    gen_stats = report._compute_generation_stats(evaluations, latest_gen=0)
    best = report._select_best(genomes, evaluations)
    inheritance_summary = report._compute_inheritance_summary(evaluations)
    family_survival = report._compute_family_survival(evaluations, genomes)

    assert summary == {
        "avg_quality": 0.9,
        "avg_train_seconds": 0.3,
        "avg_parameter_count": 120.0,
        "quality_per_second": 3.0,
        "quality_per_kparam": 0.9,
    }
    assert family_rows == [
        {
            "family": "mlp",
            "avg_quality": 0.9,
            "avg_train_seconds": 0.3,
            "avg_parameter_count": 120.0,
            "quality_per_second": 3.0,
            "quality_per_kparam": 0.9,
        }
    ]
    assert operator_rows == [
        {
            "operator": "mutation:width",
            "family": "mlp",
            "avg_quality": 0.9,
            "avg_train_seconds": 0.3,
            "avg_parameter_count": 120.0,
            "quality_per_second": 3.0,
            "quality_per_kparam": 0.9,
        }
    ]
    assert operator_success_rows == [
        {
            "operator": "mutation:width",
            "family": "mlp",
            "benchmark": "moons",
            "avg_quality": 0.9,
            "count": 1,
        }
    ]
    assert gen_stats == {0: {"best": 0.9, "avg": 0.9, "count": 1}}
    assert inheritance_summary is not None
    assert inheritance_summary["hits"] == 1
    assert inheritance_summary["rate"] == 100.0
    assert inheritance_summary["avg_quality_hit"] == 0.9
    assert math.isnan(inheritance_summary["avg_quality_miss"])
    assert family_survival == [
        {
            "generation": 0,
            "active_families": 1,
            "breakdown": "mlp(1)",
        }
    ]
    assert best is not None
    assert best.genome_id == "genome-ok"


def test_inspect_status_mix_falls_back_to_failure_reason_when_status_is_missing(monkeypatch, tmp_path):
    run_dir = tmp_path / "inspect-status-fallback"
    run_dir.mkdir()

    class FakeRunStore:
        def __init__(self, _db_path):
            pass

        def latest_generation(self, _run_id):
            return 0

        def load_evaluations(self, _run_id):
            return [
                {"benchmark_id": "moons", "failure_reason": None},
                {"benchmark_id": "digits", "failure_reason": "compile_timeout:mlx"},
            ]

        def load_genomes(self, _run_id):
            return []

        def load_best_per_benchmark(self, _run_id):
            return {"moons": {"quality": 0.91, "metric_name": "accuracy", "parameter_count": 12}}

        def close(self):
            return None

    monkeypatch.setattr("prism.storage.RunStore", FakeRunStore)
    monkeypatch.setattr("prism.export.report._resolve_run_id", lambda _store: "run-1")
    monkeypatch.setattr(
        "prism.export.report._load_runtime_metadata",
        lambda _run_dir: {
            "runtime_backend": "unknown",
            "runtime_version": "unknown",
            "precision_mode": "fp32",
            "wall_clock_seconds": 12.5,
        },
    )

    result = runner.invoke(app, ["inspect", str(run_dir)])

    assert result.exit_code == 0, result.stdout
    assert "Evaluation Status Mix" in result.stdout
    assert "ok=1, failed=1" in result.stdout
    assert "Wall Clock Seconds" in result.stdout
    assert "12.500" in result.stdout
    assert "Failure Patterns" in result.stdout
    assert "compile_timeout:mlx" in result.stdout
    assert "Recent Failures" in result.stdout
    assert "digits" in result.stdout


def test_inspect_counts_and_lists_non_ok_status_rows(monkeypatch, tmp_path):
    run_dir = tmp_path / "inspect-status-mix"
    run_dir.mkdir()

    class FakeRunStore:
        def __init__(self, _db_path):
            pass

        def latest_generation(self, _run_id):
            return 0

        def load_evaluations(self, _run_id):
            return [
                {"benchmark_id": "moons", "status": "ok", "failure_reason": None},
                {"benchmark_id": "iris", "status": "missing", "failure_reason": None},
                {"benchmark_id": "digits", "status": "failed", "failure_reason": "compile_timeout:mlx"},
            ]

        def load_genomes(self, _run_id):
            return []

        def load_best_per_benchmark(self, _run_id):
            return {"moons": {"quality": 0.91, "metric_name": "accuracy", "parameter_count": 12}}

        def close(self):
            return None

    monkeypatch.setattr("prism.storage.RunStore", FakeRunStore)
    monkeypatch.setattr("prism.export.report._resolve_run_id", lambda _store: "run-1")
    monkeypatch.setattr(
        "prism.export.report._load_runtime_metadata",
        lambda _run_dir: {
            "runtime_backend": "unknown",
            "runtime_version": "unknown",
            "precision_mode": "fp32",
            "wall_clock_seconds": 12.5,
        },
    )

    result = runner.invoke(app, ["inspect", str(run_dir)])

    assert result.exit_code == 0, result.stdout
    assert "Evaluation Status Mix" in result.stdout
    assert "ok=1, failed=1, missing=1" in result.stdout
    assert "Wall Clock Seconds" in result.stdout
    assert "12.500" in result.stdout
    assert "Failure Patterns" in result.stdout
    assert "missing" in result.stdout
    assert "compile_timeout:mlx" in result.stdout
    assert "Recent Failures" in result.stdout
    assert "iris" in result.stdout
    assert "digits" in result.stdout
