from evonn_shared.manifests import summary_core_from_results


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
