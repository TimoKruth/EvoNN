"""Markdown rendering for head-to-head comparisons."""

from __future__ import annotations

from evonn_compare.adapters.slots import system_display_name
from evonn_compare.comparison.engine import ComparisonResult


def render_comparison_markdown(result: ComparisonResult) -> str:
    """Render a markdown report for one comparison result."""

    lines = [
        f"# Symbiosis Comparison: {result.pack_name}",
        "",
        f"- Parity status: `{result.parity_status}`",
    ]
    for reason in result.reasons:
        lines.append(f"- Note: {reason}")
    lines.extend(
        [
            "",
            "## Run Telemetry",
            "",
            "| System | Run ID | Evals | Eff Epochs | QD | Novelty w | Novelty mean | Niches | Fill | Archive Parents |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
            _telemetry_row(result.left_manifest),
            _telemetry_row(result.right_manifest),
        ]
    )
    lines.extend(
        [
            "",
            "## Matchups",
            "",
            "| Benchmark | Metric | Left | Right | Winner |",
            "|---|---|---:|---:|---|",
        ]
    )
    for matchup in result.matchups:
        left_value = "---" if matchup.left_value is None else f"{matchup.left_value:.6f}"
        right_value = "---" if matchup.right_value is None else f"{matchup.right_value:.6f}"
        lines.append(
            f"| {matchup.benchmark_id} | {matchup.metric_name} | "
            f"{left_value} | {right_value} | {matchup.winner} |"
        )

    summary = result.summary
    left_label = _display_system_name(result.left_manifest.system)
    right_label = _display_system_name(result.right_manifest.system)
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- {left_label} wins: `{summary.left_wins}`",
            f"- {right_label} wins: `{summary.right_wins}`",
            f"- Ties: `{summary.ties}`",
            f"- Unsupported: `{summary.unsupported}`",
            f"- Skipped: `{summary.skipped}`",
            f"- Failed: `{summary.failed}`",
        ]
    )
    if result.matchups:
        lines.extend(
            [
                "",
                "Wilcoxon note: campaign summaries use the paired Wilcoxon test to",
                "measure how consistently deltas favor one system across benchmarks.",
                "Smaller `Wilcoxon p` means stronger evidence of a real paired",
                "difference; the delta columns determine the direction.",
            ]
        )
    return "\n".join(lines)


def _telemetry_row(manifest) -> str:
    telemetry = manifest.search_telemetry
    effective_epochs = manifest.budget.effective_training_epochs
    qd_enabled = "---"
    novelty_weight = "---"
    novelty_mean = "---"
    niches = "---"
    fill_ratio = "---"
    archive_parents = "---"
    if telemetry is not None:
        if telemetry.effective_training_epochs is not None:
            effective_epochs = telemetry.effective_training_epochs
        qd_enabled = "yes" if telemetry.qd_enabled else "no"
        novelty_weight = _float_cell(telemetry.novelty_weight)
        novelty_mean = _float_cell(telemetry.novelty_score_mean)
        niches = _int_cell(telemetry.map_elites_occupied_niches)
        fill_ratio = _float_cell(telemetry.map_elites_fill_ratio)
        archive_parents = _int_cell(telemetry.map_elites_parent_samples)
    return (
        f"| {manifest.system} | {manifest.run_id} | {manifest.budget.evaluation_count} | "
        f"{_int_cell(effective_epochs)} | {qd_enabled} | {novelty_weight} | {novelty_mean} | "
        f"{niches} | {fill_ratio} | {archive_parents} |"
    )


def _float_cell(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{value:.6f}"


def _int_cell(value: int | None) -> str:
    if value is None:
        return "---"
    return str(value)


def _display_system_name(system: str) -> str:
    return system_display_name(system)
