from __future__ import annotations

from evonn_primordia.objectives import candidate_signature, complexity_penalty, novelty_score, search_score


def test_search_score_prefers_novel_lower_complexity_candidates() -> None:
    seen = {"mlp|mlp[64x64]"}
    record = {
        "primitive_family": "mlp",
        "architecture_summary": "mlp[32x32]",
        "parameter_count": 1024,
        "train_seconds": 0.5,
        "quality": 0.9,
        "benchmark_group": "tabular",
    }

    scores = search_score(record, seen_signatures=seen, benchmark_group="tabular")

    assert scores["search_score"] > 0.0
    assert scores["novelty_score"] > 0.0
    assert scores["complexity_penalty"] >= 0.0
    assert scores["parameter_efficiency_score"] > 0.0
    assert scores["train_time_efficiency_score"] > 0.0


def test_metric_only_selection_returns_raw_quality() -> None:
    record = {
        "primitive_family": "mlp",
        "architecture_summary": "mlp[64x64]",
        "parameter_count": 4096,
        "train_seconds": 1.0,
        "quality": 0.75,
    }

    scores = search_score(record, selection_mode="metric_only")

    assert scores["search_score"] == 0.75
    assert scores["novelty_score"] == 0.0
    assert scores["complexity_penalty"] == 0.0


def test_candidate_signature_and_novelty_distinguish_seen_candidates() -> None:
    record = {
        "primitive_family": "sparse_mlp",
        "architecture_summary": "sparse_mlp[64x64]",
    }
    signature = candidate_signature(record)

    assert signature == "sparse_mlp|sparse_mlp[64x64]"
    assert novelty_score(record, {signature}) < 1.0
    assert novelty_score(record, set()) == 1.0


def test_complexity_penalty_scales_with_parameter_count() -> None:
    small = {"parameter_count": 128, "architecture_summary": "mlp[16]", "benchmark_group": "tabular"}
    large = {"parameter_count": 16384, "architecture_summary": "mlp[128x128x128]", "benchmark_group": "tabular"}

    assert complexity_penalty(large) > complexity_penalty(small)
