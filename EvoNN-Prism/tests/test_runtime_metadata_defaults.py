from __future__ import annotations

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
