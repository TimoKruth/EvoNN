"""Markdown renderer for longitudinal fair-matrix trend rows."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from evonn_compare.comparison.fair_matrix import MatrixTrendRow
from evonn_compare.reporting.fair_matrix_stats import build_multi_seed_statistics, build_scope_run_summaries
from evonn_compare.specialization import build_engine_profiles, build_family_leaderboards


def render_fair_matrix_trend_markdown(rows: Iterable[MatrixTrendRow]) -> str:
    trend_rows = sorted(
        list(rows),
        key=lambda row: (row.system, row.benchmark_id, row.budget, row.seed, row.run_id),
    )
    if not trend_rows:
        return "# Fair Matrix Trends: empty\n\n_No trend rows found._"

    pack_names = sorted({row.pack_name for row in trend_rows})
    systems = sorted({row.system for row in trend_rows})
    budgets = sorted({row.budget for row in trend_rows})
    seeds = sorted({row.seed for row in trend_rows})
    scopes = sorted({row.matrix_scope for row in trend_rows})
    comparison_labels = sorted({_comparison_label(row) for row in trend_rows})
    comparison_cohorts = sorted({_comparison_cohort(row) for row in trend_rows})
    seed_buckets = sorted({_seed_bucket(row) for row in trend_rows})
    benchmark_families = sorted({row.benchmark_family for row in trend_rows})
    lane_states = sorted({row.lane_operating_state for row in trend_rows})
    accounting_states = sorted({"ok" if row.lane_budget_accounting_ok else "incomplete" for row in trend_rows})
    repeatability_states = sorted({"ready" if row.lane_repeatability_ready else "not-ready" for row in trend_rows})
    unique_lane_runs = sorted({(_comparison_label(row), _comparison_case_id(row)) for row in trend_rows})

    lines = [
        f"# Fair Matrix Trends: {pack_names[0]}",
        "",
        "## Trend Dataset Summary",
        "",
        f"- Packs: `{', '.join(pack_names)}`",
        f"- Systems: `{', '.join(systems)}`",
        f"- Budgets: `{', '.join(str(value) for value in budgets)}`",
        f"- Seeds: `{', '.join(str(value) for value in seeds)}`",
        f"- Fairness Scope: `{', '.join(scopes)}`",
        f"- Comparison Labels: `{', '.join(comparison_labels)}`",
        f"- Comparison Cohorts: `{', '.join(comparison_cohorts)}`",
        f"- Seeding Buckets: `{', '.join(seed_buckets)}`",
        f"- Benchmark Families: `{', '.join(benchmark_families)}`",
        f"- Lane States: `{', '.join(lane_states)}`",
        f"- Budget Accounting: `{', '.join(accounting_states)}`",
        f"- Repeatability: `{', '.join(repeatability_states)}`",
        f"- Rows: `{len(trend_rows)}`",
        f"- Unique Lane Runs: `{len(unique_lane_runs)}`",
        "",
        "## Lane Health By Budget",
        "",
        "| Comparison | Budget | Runs | Lane States | Accounting | Repeatability |",
        "|---|---:|---:|---|---|---|",
    ]

    budget_rows: dict[tuple[str, int], list[MatrixTrendRow]] = defaultdict(list)
    for row in trend_rows:
        budget_rows[(_comparison_label(row), row.budget)].append(row)
    for comparison_label, budget in sorted(budget_rows):
        entries = budget_rows[(comparison_label, budget)]
        run_count = len({_comparison_case_id(row) for row in entries})
        budget_lane_states = ", ".join(sorted({row.lane_operating_state for row in entries}))
        budget_accounting_states = ", ".join(sorted({"ok" if row.lane_budget_accounting_ok else "incomplete" for row in entries}))
        budget_repeatability_states = ", ".join(sorted({"ready" if row.lane_repeatability_ready else "not-ready" for row in entries}))
        lines.append(
            f"| {comparison_label} | {budget} | {run_count} | {budget_lane_states} | {budget_accounting_states} | {budget_repeatability_states} |"
        )

    seed_snapshots = build_scope_run_summaries(trend_rows, systems=tuple(systems))
    lines.extend(
        [
            "",
            "## Per-Seed Aggregate Snapshots",
            "",
            "| Comparison | Pack | Budget | Seed | Lane State | Repeatability | Accounting | System Scores | Ties | Skipped |",
            "|---|---|---:|---:|---|---|---|---|---:|---:|",
        ]
    )
    for run in seed_snapshots:
        scope_rows = sorted(run["scope"]["rows"], key=lambda row: row["system"])
        score_summary = "; ".join(
            (
                f"{row['system']}={_float_cell(float(row['solo_wins']) + 0.5 * float(row['shared_wins']))} "
                f"({row['solo_wins']} solo/{row['shared_wins']} shared)"
            )
            for row in scope_rows
        )
        lines.append(
            f"| {run['comparison_label']} | {run['pack_name']} | {run['budget']} | {run['seed']} | {run['lane_operating_state']} | "
            f"{'ready' if run['repeatability_ready'] else 'not-ready'} | "
            f"{'ok' if run['budget_accounting_ok'] else 'incomplete'} | {score_summary} | "
            f"{run['scope']['ties']} | {run['scope']['skipped']} |"
        )

    multi_seed_groups = build_multi_seed_statistics(trend_rows, systems=tuple(systems))
    lines.extend(
        [
            "",
            "## Multi-Seed Statistical Summary",
            "",
            "| Comparison | Pack | Budget | Seeds | Seed IDs | System | Mean Score | Score SD | Score Range | Best | Worst | 95% CI | Mean Solo Wins | Mean Shared Wins | Mean Failures | Mean Missing |",
            "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for group in multi_seed_groups:
        ordered_rows = sorted(group["system_rows"], key=lambda row: (-float(row["mean_score"]), row["system"]))
        seed_ids = ", ".join(str(seed) for seed in group["seeds"])
        for row in ordered_rows:
            lines.append(
                f"| {group['comparison_label']} | {group['pack_name']} | {group['budget']} | {group['seed_count']} | {seed_ids} | {row['system']} | "
                f"{_float_cell(row['mean_score'])} | {_float_cell(row['score_stddev'])} | "
                f"{_float_cell(row['score_range'])} | {_float_cell(row['best_score'])} | {_float_cell(row['worst_score'])} | "
                f"{_ci_cell(row['score_ci95_low'], row['score_ci95_high'])} | "
                f"{_float_cell(row['mean_solo_wins'])} | {_float_cell(row['mean_shared_wins'])} | "
                f"{_float_cell(row['mean_benchmark_failures'])} | {_float_cell(row['mean_missing_results'])} |"
            )

    lines.extend(
        [
            "",
            "## Multi-Seed Pairwise Deltas",
            "",
            "| Comparison | Pack | Budget | Pair | Seeds | Left Better | Ties | Right Better | Mean Delta | Delta SD | 95% CI | Sign Test p |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for group in multi_seed_groups:
        for row in group["pairwise"]:
            lines.append(
                f"| {group['comparison_label']} | {group['pack_name']} | {group['budget']} | {row['left_system']} vs {row['right_system']} | "
                f"{row['seed_count']} | {row['left_better']} | {row['ties']} | {row['right_better']} | "
                f"{_float_cell(row['mean_score_delta'])} | {_float_cell(row['score_delta_stddev'])} | "
                f"{_ci_cell(row['score_delta_ci95_low'], row['score_delta_ci95_high'])} | "
                f"{_float_cell(row['sign_test_p_value'])} |"
            )

    family_leaderboards = build_family_leaderboards(trend_rows, systems=tuple(systems))
    lines.extend(
        [
            "",
            "## Benchmark Family Leaderboards",
            "",
            "| Comparison | Pack | Budget | Family | Benchmarks | Seeds | Rank | System | Mean Score | Score | Solo Wins | Shared Wins | Failures | Missing |",
            "|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in family_leaderboards:
        for row in group["system_rows"]:
            lines.append(
                f"| {group['comparison_label']} | {group['pack_name']} | {group['budget']} | {group['benchmark_family']} | "
                f"{group['benchmark_count']} | {group['seed_count']} | {row['rank']} | {row['system']} | "
                f"{_float_cell(row['mean_score'])} | {_float_cell(row['score'])} | {row['solo_wins']} | "
                f"{row['shared_wins']} | {row['benchmark_failures']} | {row['missing_results']} |"
            )

    lines.extend(
        [
            "",
            "## Outcome Status by System",
            "",
            "| System | ok | failed | skipped | unsupported | missing |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in trend_rows:
        status_counts[row.system][row.outcome_status] += 1
    for system in systems:
        counts = status_counts[system]
        lines.append(
            f"| {system} | {counts.get('ok', 0)} | {counts.get('failed', 0)} | "
            f"{counts.get('skipped', 0)} | {counts.get('unsupported', 0)} | {counts.get('missing', 0)} |"
        )

    engine_profiles = build_engine_profiles(trend_rows, systems=tuple(systems))
    lines.extend(
        [
            "",
            "## Engine Profile Reports",
        ]
    )
    for profile in engine_profiles:
        lines.extend(
            [
                "",
                f"### {_titleize(profile['system'])} [{profile['comparison_label']}]",
                "",
                f"- Comparison Cohort: `{profile['comparison_cohort']}`",
                f"- Search Style: `{profile['search_style']}`",
                f"- Expected Evidence: {profile['expected_signal']}",
                f"- Branch Review Lens: {profile['branch_review_prompt']}",
                f"- Status Counts: ok={profile['status_counts']['ok']}, failed={profile['status_counts']['failed']}, skipped={profile['status_counts']['skipped']}, unsupported={profile['status_counts']['unsupported']}, missing={profile['status_counts']['missing']}",
                f"- Strongest Families: {_family_observations(profile['family_strengths'])}",
                f"- Weakest Families: {_family_observations(profile['family_weaknesses'])}",
                f"- Recurring Failure Patterns: {_failure_patterns(profile['failure_patterns'])}",
                f"- Architecture Signals: {_architecture_examples(profile['architecture_examples'])}",
            ]
        )

    lines.extend(
        [
            "",
            "## Benchmark Trend View",
            "",
            "| Comparison | System | Benchmark | Runs | Latest | Best | Delta | Latest Status | Budget | Seed | Scope | Seed Bucket | Seed Source | Lane State | Accounting | Repeatability | System State |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|---|---|",
        ]
    )

    grouped: dict[tuple[str, str, str], list[MatrixTrendRow]] = defaultdict(list)
    for row in trend_rows:
        grouped[(_comparison_label(row), row.system, row.benchmark_id)].append(row)
    for (comparison_label, system, benchmark_id), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: (row.budget, row.seed, row.run_id))
        latest = ordered[-1]
        metric_values = [row.metric_value for row in ordered if row.metric_value is not None]
        latest_metric = _float_cell(latest.metric_value)
        best_metric = _float_cell(max(metric_values) if metric_values else None)
        delta_value = None
        if len(metric_values) >= 2:
            delta_value = metric_values[-1] - metric_values[0]
        else:
            delta_value = 0.0 if metric_values else None
        lines.append(
            f"| {comparison_label} | {system} | {benchmark_id} | {len(ordered)} | {latest_metric} | {best_metric} | "
            f"{_float_cell(delta_value)} | {latest.outcome_status} | {latest.budget} | {latest.seed} | {latest.matrix_scope} | "
            f"{_seed_bucket(latest)} | {_seed_source(latest)} | "
            f"{latest.lane_operating_state} | {'ok' if latest.lane_budget_accounting_ok else 'incomplete'} | "
            f"{'ready' if latest.lane_repeatability_ready else 'not-ready'} | {latest.system_operating_state} |"
        )

    return "\n".join(lines)


def _float_cell(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{float(value):.6f}"


def _seed_bucket(row: MatrixTrendRow) -> str:
    return str(row.fairness_metadata.get("seeding_bucket") or "transfer-opaque")


def _seed_source(row: MatrixTrendRow) -> str:
    source_system = row.fairness_metadata.get("seed_source_system")
    source_run_id = row.fairness_metadata.get("seed_source_run_id")
    if not source_system:
        return "---"
    if source_run_id:
        return f"{source_system}:{source_run_id}"
    return str(source_system)


def _comparison_label(row: MatrixTrendRow) -> str:
    return str(row.fairness_metadata.get("comparison_label") or row.fairness_metadata.get("baseline_label") or row.fairness_metadata.get("comparison_cohort") or "current-workspace")


def _comparison_cohort(row: MatrixTrendRow) -> str:
    return str(row.fairness_metadata.get("comparison_cohort") or "current-workspace")


def _comparison_case_id(row: MatrixTrendRow) -> str:
    return str(row.fairness_metadata.get("comparison_case_id") or f"{_comparison_label(row)}:{row.pack_name}:{row.budget}:{row.seed}")


def _ci_cell(lower: float, upper: float) -> str:
    return f"{lower:.6f} to {upper:.6f}"


def _titleize(system: str) -> str:
    return system.replace("_", " ").title()


def _family_observations(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "none"
    return "; ".join(
        (
            f"{row['benchmark_family']} @ {row['pack_name']}:{row['budget']} "
            f"(rank {row['rank']}/{row['system_count']}, mean {_float_cell(float(row['mean_score']))}, "
            f"fail {row['benchmark_failures']}, missing {row['missing_results']})"
        )
        for row in rows
    )


def _failure_patterns(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "none"
    return "; ".join(f"{row['reason']} ({row['count']})" for row in rows)


def _architecture_examples(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "none"
    return "; ".join(f"{row['summary']} ({row['count']})" for row in rows)
