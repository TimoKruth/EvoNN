from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.portable_smoke import (
    _load_benchmark_arrays,
    ensure_prism_portable_smoke_export,
    ensure_topograph_portable_smoke_export,
)


def _write_pack(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "portable_smoke",
                "tier": 1,
                "description": "portable smoke",
                "benchmarks": [
                    {
                        "benchmark_id": "iris_classification",
                        "native_ids": {"evonn": "iris_classification", "evonn2": "iris"},
                        "task_kind": "classification",
                        "metric_name": "accuracy",
                        "metric_direction": "max",
                    },
                    {
                        "benchmark_id": "diabetes_regression",
                        "native_ids": {"evonn": "diabetes_regression", "evonn2": "diabetes"},
                        "task_kind": "regression",
                        "metric_name": "mse",
                        "metric_direction": "min",
                    },
                ],
                "budget_policy": {
                    "evaluation_count": 4,
                    "epochs_per_candidate": 1,
                    "budget_tolerance_pct": 10.0,
                },
                "seed_policy": {"mode": "campaign", "required": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_portable_prism_smoke_export_produces_valid_compare_contract(tmp_path: Path) -> None:
    pack_path = _write_pack(tmp_path / "pack.yaml")
    config_path = tmp_path / "prism.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "benchmark_pack": {
                    "pack_name": str(pack_path),
                    "benchmark_ids": ["iris_classification", "diabetes_regression"],
                },
                "training": {"epochs": 1},
                "evolution": {
                    "population_size": 2,
                    "num_generations": 1,
                    "allowed_families": ["mlp", "sparse_mlp"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "prism-run"

    ensure_prism_portable_smoke_export(
        config_path=config_path,
        pack_path=pack_path,
        run_dir=run_dir,
        output_dir=run_dir,
    )

    pack = load_parity_pack(pack_path)
    report = SystemIngestor(run_dir).validate(pack)

    assert report.ok, report.model_dump()
    manifest = SystemIngestor(run_dir).load_manifest()
    results = SystemIngestor(run_dir).load_results()
    assert manifest.system == "prism"
    assert manifest.device.framework == "portable-sklearn"
    assert manifest.budget.actual_evaluations == 4
    assert manifest.budget.cached_evaluations == 0
    assert manifest.budget.failed_evaluations == 0
    assert manifest.budget.invalid_evaluations == 0
    assert manifest.budget.partial_run is False
    assert "portable candidate-benchmark trial" in manifest.budget.evaluation_semantics
    assert all(row.status == "ok" for row in results)
    assert (run_dir / "summary.json").exists()


def test_portable_topograph_smoke_export_produces_valid_compare_contract(tmp_path: Path) -> None:
    pack_path = _write_pack(tmp_path / "pack.yaml")
    config_path = tmp_path / "topograph.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "benchmark": "iris",
                "benchmark_pool": {"benchmarks": ["iris", "diabetes"], "sample_k": 2},
                "training": {"epochs": 1},
                "evolution": {"population_size": 2, "num_generations": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "topograph-run"

    ensure_topograph_portable_smoke_export(
        config_path=config_path,
        pack_path=pack_path,
        run_dir=run_dir,
        output_dir=run_dir,
    )

    pack = load_parity_pack(pack_path)
    report = SystemIngestor(run_dir).validate(pack)

    assert report.ok, report.model_dump()
    manifest = SystemIngestor(run_dir).load_manifest()
    results = SystemIngestor(run_dir).load_results()
    assert manifest.system == "topograph"
    assert manifest.device.framework == "portable-sklearn"
    assert manifest.budget.actual_evaluations == 4
    assert manifest.budget.cached_evaluations == 0
    assert manifest.budget.failed_evaluations == 0
    assert manifest.budget.invalid_evaluations == 0
    assert manifest.budget.partial_run is False
    assert "portable candidate-benchmark trial" in manifest.budget.evaluation_semantics
    assert all(row.status == "ok" for row in results)
    assert (run_dir / "summary.json").exists()


def test_portable_topograph_smoke_export_surfaces_primordia_seeding(tmp_path: Path) -> None:
    pack_path = _write_pack(tmp_path / "pack.yaml")
    seed_candidates_path = tmp_path / "seed_candidates.json"
    seed_candidates_path.write_text(
        json.dumps(
            {
                "system": "primordia",
                "run_id": "prim-run-7",
                "seed_candidates": [
                    {
                        "seed_rank": 1,
                        "family": "mlp",
                        "benchmark_groups": ["tabular"],
                        "representative_genome_id": "seeded-genome",
                        "representative_architecture_summary": "mlp[64x64]",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "topograph.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "benchmark": "iris",
                "benchmark_pool": {
                    "benchmarks": ["iris", "diabetes"],
                    "sample_k": 2,
                    "primordia_seed_candidates_path": str(seed_candidates_path),
                },
                "training": {"epochs": 1},
                "evolution": {"population_size": 2, "num_generations": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "topograph-run"

    ensure_topograph_portable_smoke_export(
        config_path=config_path,
        pack_path=pack_path,
        run_dir=run_dir,
        output_dir=run_dir,
    )

    manifest = SystemIngestor(run_dir).load_manifest()

    assert manifest.seeding is not None
    assert manifest.seeding.seeding_ladder == "direct"
    assert manifest.seeding.seed_source_system == "primordia"
    assert manifest.seeding.seed_source_run_id == "prim-run-7"
    assert manifest.seeding.seed_artifact_path == str(seed_candidates_path)
    assert manifest.seeding.seed_selected_family == "mlp"
    assert manifest.seeding.representative_architecture_summary == "mlp[64x64]"


def test_load_benchmark_arrays_preserves_original_loader_error_for_unknown_fallbacks() -> None:
    class BrokenSpec:
        def load_data(self, *, seed: int):
            raise ValueError(f"boom-{seed}")

    with pytest.raises(ValueError, match="boom-7"):
        _load_benchmark_arrays(BrokenSpec(), benchmark_id="unknown_regression", seed=7)
