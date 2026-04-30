from evonn_shared.manifests import legacy_topograph_primordia_seeding_manifest, summary_core_from_results


def test_summary_core_uses_successful_metric_values_for_median_quality() -> None:
    core = summary_core_from_results(
        results=[
            {
                "benchmark_id": "iris",
                "metric_value": 0.91,
                "quality": 9.1,
                "status": "ok",
            },
            {
                "benchmark_id": "moons",
                "metric_value": 0.87,
                "quality": 8.7,
                "status": "ok",
            },
            {
                "benchmark_id": "broken",
                "metric_value": None,
                "quality": -999.0,
                "status": "failed",
                "failure_reason": "boom",
            },
        ],
        parameter_counts=[64, 128],
    )

    assert core["best_fitness"] == {"iris": 0.91, "moons": 0.87}
    assert core["median_parameter_count"] == 96
    assert core["median_benchmark_quality"] == 0.89
    assert core["failure_count"] == 1
    assert core["failure_patterns"] == {"boom": 1}
    assert core["benchmarks_evaluated"] == 2


def test_legacy_topograph_primordia_seeding_manifest_normalizes_payload() -> None:
    manifest = legacy_topograph_primordia_seeding_manifest(
        {
            "seed_path": "/tmp/primordia/seed_candidates.json",
            "target_family": "tabular",
            "selected_family": "sparse_mlp",
            "selected_rank": 2,
            "seed_source_run_id": "prim-run-7",
        }
    )

    assert manifest is not None
    assert manifest["seeding_enabled"] is True
    assert manifest["seeding_ladder"] == "direct"
    assert manifest["seed_source_system"] == "primordia"
    assert manifest["seed_source_run_id"] == "prim-run-7"
    assert manifest["seed_artifact_path"] == "/tmp/primordia/seed_candidates.json"
    assert manifest["seed_target_family"] == "tabular"
    assert manifest["seed_selected_family"] == "sparse_mlp"
    assert manifest["seed_rank"] == 2
    assert manifest["seed_overlap_policy"] == "family-overlapping"


def test_legacy_topograph_primordia_seeding_manifest_returns_none_without_seed_artifact_path() -> None:
    manifest = legacy_topograph_primordia_seeding_manifest(
        {
            "target_family": "tabular",
            "selected_family": "sparse_mlp",
            "selected_rank": 2,
            "seed_source_run_id": "prim-run-7",
        }
    )

    assert manifest is None


def test_legacy_topograph_primordia_seeding_manifest_treats_blank_seed_artifact_path_as_missing() -> None:
    manifest = legacy_topograph_primordia_seeding_manifest(
        {
            "seed_path": "   ",
            "target_family": "tabular",
            "selected_family": "sparse_mlp",
        }
    )

    assert manifest is None
