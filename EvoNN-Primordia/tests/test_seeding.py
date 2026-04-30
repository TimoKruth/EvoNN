from __future__ import annotations

from evonn_primordia.export.seeding import build_seed_candidates


def test_seed_candidates_include_provenance_and_repeat_support() -> None:
    summary = {
        "run_id": "seed-run",
        "run_name": "seed-run",
        "runtime": "numpy-fallback",
        "runtime_version": "fallback-1.0",
        "primitive_usage": {"mlp": 3, "sparse_mlp": 2},
    }
    best_results = [
        {
            "benchmark_name": "iris",
            "benchmark_group": "tabular",
            "primitive_family": "mlp",
            "metric_name": "accuracy",
            "metric_value": 0.9,
            "quality": 0.9,
            "status": "ok",
            "genome_id": "g-1",
            "architecture_summary": "mlp[64x64]",
        },
        {
            "benchmark_name": "wine",
            "benchmark_group": "tabular",
            "primitive_family": "sparse_mlp",
            "metric_name": "accuracy",
            "metric_value": 0.88,
            "quality": 0.88,
            "status": "ok",
            "genome_id": "g-2",
            "architecture_summary": "sparse_mlp[64x64]",
        },
    ]
    trial_records = [
        {**best_results[0]},
        {**best_results[0], "benchmark_name": "breast_cancer", "quality": 0.91},
        {**best_results[1]},
    ]

    payload = build_seed_candidates(summary=summary, best_results=best_results, trial_records=trial_records)
    top = payload["seed_candidates"][0]

    assert "supporting_benchmarks" in top
    assert "repeat_support_count" in top
    assert "median_quality" in top
    assert "median_quality_by_group" in top
    assert top["repeat_support_count"] >= 1
