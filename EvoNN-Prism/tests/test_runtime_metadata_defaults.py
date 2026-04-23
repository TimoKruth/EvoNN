from __future__ import annotations

from prism.export import report
from prism.export import symbiosis as sym


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
