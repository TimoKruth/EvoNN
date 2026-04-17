"""Markdown renderer for campaign summaries."""

from __future__ import annotations

from evonn_compare.comparison.leaderboard import CampaignLeaderboard


def render_campaign_markdown(leaderboard: CampaignLeaderboard) -> str:
    """Render a compact markdown summary for a completed campaign."""

    lines = [
        f"# Symbiosis Campaign: {leaderboard.pack_name}",
        "",
        f"- Comparisons: `{len(leaderboard.records)}`",
        "",
        "## Budget Summary",
        "",
        "| Budget | Pairs | EvoNN pair wins | EvoNN-2 pair wins | Tied pairs | EvoNN benchmark wins | EvoNN-2 benchmark wins | Benchmark ties | Sign-test p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard.rows:
        p_value = "---" if row.sign_test_p_value is None else f"{row.sign_test_p_value:.6f}"
        lines.append(
            f"| {row.budget} | {row.pair_count} | {row.evonn_pair_wins} | "
            f"{row.evonn2_pair_wins} | {row.tied_pairs} | {row.evonn_benchmark_wins} | "
            f"{row.evonn2_benchmark_wins} | {row.benchmark_ties} | {p_value} |"
        )

    if leaderboard.rows:
        lines.extend(
            [
                "",
                "## Statistical Analysis",
                "",
                "| Budget | Wilcoxon p | Mean Delta | Median Delta | 95% CI |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in leaderboard.rows:
            wilcoxon_p = "---" if row.wilcoxon_p_value is None else f"{row.wilcoxon_p_value:.6f}"
            mean_delta = "---" if row.mean_delta is None else f"{row.mean_delta:.6f}"
            median_delta = "---" if row.median_delta is None else f"{row.median_delta:.6f}"
            if row.bootstrap_ci_lower is None or row.bootstrap_ci_upper is None:
                ci = "---"
            else:
                ci = f"[{row.bootstrap_ci_lower:.6f}, {row.bootstrap_ci_upper:.6f}]"
            lines.append(
                f"| {row.budget} | {wilcoxon_p} | {mean_delta} | {median_delta} | {ci} |"
            )
        lines.extend(
            [
                "",
                "Wilcoxon note: smaller `Wilcoxon p` means stronger evidence of a",
                "consistent paired difference across benchmarks. Read it together with",
                "`Mean Delta` and `Median Delta` to determine which system is ahead.",
            ]
        )
        if any(
            row.evonn2_effective_training_epochs_mean is not None
            or row.evonn2_map_elites_occupied_niches_mean is not None
            for row in leaderboard.rows
        ):
            lines.extend(
                [
                    "",
                    "## EvoNN-2 Search Telemetry",
                    "",
                    "| Budget | Novelty w | Archive Ratio | Eff Epochs | Novelty Archive | Novelty Mean | Niches | Fill | Archive Parents |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in leaderboard.rows:
                lines.append(
                    f"| {row.budget} | {_float_cell(row.evonn2_novelty_weight_mean)} | "
                    f"{_float_cell(row.evonn2_map_elites_selection_ratio_mean)} | "
                    f"{_float_cell(row.evonn2_effective_training_epochs_mean)} | "
                    f"{_float_cell(row.evonn2_novelty_archive_final_size_mean)} | "
                    f"{_float_cell(row.evonn2_novelty_score_mean_mean)} | "
                    f"{_float_cell(row.evonn2_map_elites_occupied_niches_mean)} | "
                    f"{_float_cell(row.evonn2_map_elites_fill_ratio_mean)} | "
                    f"{_float_cell(row.evonn2_map_elites_parent_samples_mean)} |"
                )

    if leaderboard.records:
        lines.extend(
            [
                "",
                "## Cases",
                "",
                "| Budget | Seed | Parity | EvoNN wins | EvoNN-2 wins | Ties | Report |",
                "|---:|---:|---|---:|---:|---:|---|",
            ]
        )
        for record in sorted(leaderboard.records, key=lambda item: (item.budget, item.seed)):
            lines.append(
                f"| {record.budget} | {record.seed} | {record.parity_status} | "
                f"{record.evonn_wins} | {record.evonn2_wins} | {record.ties} | "
                f"{record.comparison_report} |"
            )

    return "\n".join(lines)


def _float_cell(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{value:.6f}"
