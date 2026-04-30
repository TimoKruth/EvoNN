"""Markdown rendering for head-to-head comparisons."""

from __future__ import annotations

from evonn_compare.adapters.slots import system_display_name
from evonn_compare.comparison.engine import ComparisonResult


def render_comparison_markdown(result: ComparisonResult) -> str:
    """Render a markdown report for one comparison result."""

    left_label = _summary_label(result.left_manifest, result.right_manifest)
    right_label = _summary_label(result.right_manifest, result.left_manifest)
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
            "| System | Run ID | Evals | Policy | Data Sig | Code | Seed Mode | Seed Source | Seed Artifact | Seed Family | Eff Epochs | QD | Novelty w | Novelty mean | Niches | Fill | Archive Parents |",
            "|---|---|---:|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
            _telemetry_row(result.left_manifest, system_label=left_label),
            _telemetry_row(result.right_manifest, system_label=right_label),
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


def _telemetry_row(manifest, *, system_label: str | None = None) -> str:
    telemetry = manifest.search_telemetry
    fairness = manifest.fairness
    effective_epochs = manifest.budget.effective_training_epochs
    budget_policy = manifest.budget.budget_policy_name or "---"
    data_signature = fairness.data_signature if fairness is not None else "---"
    code_version = fairness.code_version[:12] if fairness is not None and fairness.code_version else "---"
    seed_mode = _seed_mode(manifest)
    seed_source = _seed_source(manifest)
    seed_artifact = _seed_artifact(manifest)
    seed_family = _seed_family(manifest)
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
    display_name = system_label or manifest.system
    return (
        f"| {display_name} | {manifest.run_id} | {manifest.budget.evaluation_count} | "
        f"{budget_policy} | {data_signature} | {code_version} | {seed_mode} | {seed_source} | "
        f"{seed_artifact} | {seed_family} | {_int_cell(effective_epochs)} | "
        f"{qd_enabled} | {novelty_weight} | {novelty_mean} | "
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


def _summary_label(manifest, other_manifest) -> str:
    label = _display_system_name(manifest.system)
    if manifest.system != other_manifest.system:
        return label
    seed_mode = _seed_mode(manifest)
    other_seed_mode = _seed_mode(other_manifest)
    if seed_mode == "transfer-opaque" and other_seed_mode != "transfer-opaque":
        seed_mode = "unseeded"
    if seed_mode != other_seed_mode:
        return f"{label} ({seed_mode})"
    if manifest.run_id != other_manifest.run_id:
        return f"{label} ({manifest.run_id})"
    return label


def _seed_mode(manifest) -> str:
    seeding = manifest.seeding
    if seeding is None:
        return "transfer-opaque"
    if not seeding.seeding_enabled or seeding.seeding_ladder == "none":
        return "unseeded"
    return seeding.seeding_ladder


def _seed_source(manifest) -> str:
    seeding = manifest.seeding
    if seeding is None or seeding.seed_source_system is None:
        return "---"
    if seeding.seed_source_run_id:
        return f"{seeding.seed_source_system}:{seeding.seed_source_run_id}"
    return seeding.seed_source_system


def _seed_artifact(manifest) -> str:
    seeding = manifest.seeding
    if seeding is None or seeding.seed_artifact_path is None:
        return "---"
    return seeding.seed_artifact_path


def _seed_family(manifest) -> str:
    seeding = manifest.seeding
    if seeding is None:
        return "---"
    if seeding.seed_selected_family and seeding.seed_target_family:
        return f"{seeding.seed_selected_family}->{seeding.seed_target_family}"
    return seeding.seed_selected_family or seeding.seed_target_family or "---"
