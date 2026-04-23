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
