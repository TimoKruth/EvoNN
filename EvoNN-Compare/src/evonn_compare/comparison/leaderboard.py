"""Campaign-level aggregation for repeated compare runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evonn_compare.comparison.statistics import two_sided_sign_test
from evonn_compare.comparison.wilcoxon import bootstrap_confidence_interval, wilcoxon_signed_rank
from evonn_compare.contracts.models import SearchTelemetry


@dataclass(frozen=True)
class CampaignRecord:
    """One completed comparison in a campaign matrix."""

    seed: int
    budget: int
    pack_name: str
    evonn_run_id: str
    evonn2_run_dir: Path
    comparison_report: Path
    parity_status: str
    evonn_wins: int
    evonn2_wins: int
    ties: int
    benchmark_deltas: dict[str, float]
    evonn_search_telemetry: SearchTelemetry | None = None
    evonn2_search_telemetry: SearchTelemetry | None = None


@dataclass(frozen=True)
class BudgetLeaderboardRow:
    """Per-budget aggregate of campaign results."""

    budget: int
    pair_count: int
    evonn_pair_wins: int
    evonn2_pair_wins: int
    tied_pairs: int
    evonn_benchmark_wins: int
    evonn2_benchmark_wins: int
    benchmark_ties: int
    sign_test_p_value: float | None
    wilcoxon_p_value: float | None
    median_delta: float | None
    mean_delta: float | None
    bootstrap_ci_lower: float | None
    bootstrap_ci_upper: float | None
    evonn2_novelty_weight_mean: float | None
    evonn2_map_elites_selection_ratio_mean: float | None
    evonn2_effective_training_epochs_mean: float | None
    evonn2_novelty_archive_final_size_mean: float | None
    evonn2_novelty_score_mean_mean: float | None
    evonn2_map_elites_occupied_niches_mean: float | None
    evonn2_map_elites_fill_ratio_mean: float | None
    evonn2_map_elites_parent_samples_mean: float | None


@dataclass(frozen=True)
class CampaignLeaderboard:
    """Top-level campaign summary."""

    pack_name: str
    records: list[CampaignRecord]
    rows: list[BudgetLeaderboardRow]


def build_campaign_leaderboard(
    *,
    pack_name: str,
    records: list[CampaignRecord],
) -> CampaignLeaderboard:
    grouped = sorted({record.budget for record in records})
    rows: list[BudgetLeaderboardRow] = []
    for budget in grouped:
        subset = [record for record in records if record.budget == budget]
        evonn_pair_wins = sum(1 for record in subset if record.evonn_wins > record.evonn2_wins)
        evonn2_pair_wins = sum(1 for record in subset if record.evonn2_wins > record.evonn_wins)
        tied_pairs = sum(1 for record in subset if record.evonn_wins == record.evonn2_wins)
        non_tied = evonn_pair_wins + evonn2_pair_wins
        p_value = None if non_tied == 0 else two_sided_sign_test(evonn2_pair_wins, evonn_pair_wins)
        wilcoxon_p_value = None
        median_delta = None
        mean_delta = None
        bootstrap_ci_lower = None
        bootstrap_ci_upper = None
        evonn2_novelty_weight_mean = _mean_telemetry(subset, "novelty_weight")
        evonn2_map_elites_selection_ratio_mean = _mean_telemetry(
            subset,
            "map_elites_selection_ratio",
        )
        evonn2_effective_training_epochs_mean = _mean_telemetry(
            subset,
            "effective_training_epochs",
        )
        evonn2_novelty_archive_final_size_mean = _mean_telemetry(
            subset,
            "novelty_archive_final_size",
        )
        evonn2_novelty_score_mean_mean = _mean_telemetry(subset, "novelty_score_mean")
        evonn2_map_elites_occupied_niches_mean = _mean_telemetry(
            subset,
            "map_elites_occupied_niches",
        )
        evonn2_map_elites_fill_ratio_mean = _mean_telemetry(subset, "map_elites_fill_ratio")
        evonn2_map_elites_parent_samples_mean = _mean_telemetry(
            subset,
            "map_elites_parent_samples",
        )
        averaged_deltas = _average_benchmark_deltas(subset)
        if averaged_deltas:
            stats = wilcoxon_signed_rank(averaged_deltas, [0.0] * len(averaged_deltas))
            ci = bootstrap_confidence_interval(averaged_deltas, seed=budget)
            wilcoxon_p_value = stats.p_value
            median_delta = stats.median_delta
            mean_delta = stats.mean_delta
            bootstrap_ci_lower = ci.lower
            bootstrap_ci_upper = ci.upper
        rows.append(
            BudgetLeaderboardRow(
                budget=budget,
                pair_count=len(subset),
                evonn_pair_wins=evonn_pair_wins,
                evonn2_pair_wins=evonn2_pair_wins,
                tied_pairs=tied_pairs,
                evonn_benchmark_wins=sum(record.evonn_wins for record in subset),
                evonn2_benchmark_wins=sum(record.evonn2_wins for record in subset),
                benchmark_ties=sum(record.ties for record in subset),
                sign_test_p_value=p_value,
                wilcoxon_p_value=wilcoxon_p_value,
                median_delta=median_delta,
                mean_delta=mean_delta,
                bootstrap_ci_lower=bootstrap_ci_lower,
                bootstrap_ci_upper=bootstrap_ci_upper,
                evonn2_novelty_weight_mean=evonn2_novelty_weight_mean,
                evonn2_map_elites_selection_ratio_mean=evonn2_map_elites_selection_ratio_mean,
                evonn2_effective_training_epochs_mean=evonn2_effective_training_epochs_mean,
                evonn2_novelty_archive_final_size_mean=evonn2_novelty_archive_final_size_mean,
                evonn2_novelty_score_mean_mean=evonn2_novelty_score_mean_mean,
                evonn2_map_elites_occupied_niches_mean=evonn2_map_elites_occupied_niches_mean,
                evonn2_map_elites_fill_ratio_mean=evonn2_map_elites_fill_ratio_mean,
                evonn2_map_elites_parent_samples_mean=evonn2_map_elites_parent_samples_mean,
            )
        )
    return CampaignLeaderboard(pack_name=pack_name, records=records, rows=rows)


def _average_benchmark_deltas(records: list[CampaignRecord]) -> list[float]:
    per_benchmark: dict[str, list[float]] = {}
    for record in records:
        for benchmark_id, delta in record.benchmark_deltas.items():
            per_benchmark.setdefault(benchmark_id, []).append(delta)
    return [
        sum(values) / len(values)
        for benchmark_id, values in sorted(per_benchmark.items())
        if values
    ]


def _mean_telemetry(records: list[CampaignRecord], field_name: str) -> float | None:
    values: list[float] = []
    for record in records:
        telemetry = record.evonn2_search_telemetry
        if telemetry is None:
            continue
        value = getattr(telemetry, field_name)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)
