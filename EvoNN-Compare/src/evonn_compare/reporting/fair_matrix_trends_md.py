"""Markdown renderer for longitudinal fair-matrix trend rows."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from evonn_compare.comparison.fair_matrix import MatrixTrendRow


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
    lane_states = sorted({row.lane_operating_state for row in trend_rows})
    accounting_states = sorted({"ok" if row.lane_budget_accounting_ok else "incomplete" for row in trend_rows})
    repeatability_states = sorted({"ready" if row.lane_repeatability_ready else "not-ready" for row in trend_rows})

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
        f"- Lane States: `{', '.join(lane_states)}`",
        f"- Budget Accounting: `{', '.join(accounting_states)}`",
        f"- Repeatability: `{', '.join(repeatability_states)}`",
        f"- Rows: `{len(trend_rows)}`",
        "",
        "## Outcome Status by System",
        "",
        "| System | ok | failed | skipped | unsupported | missing |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in trend_rows:
        status_counts[row.system][row.outcome_status] += 1
    for system in systems:
        counts = status_counts[system]
        lines.append(
            f"| {system} | {counts.get('ok', 0)} | {counts.get('failed', 0)} | "
            f"{counts.get('skipped', 0)} | {counts.get('unsupported', 0)} | {counts.get('missing', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Benchmark Trend View",
            "",
            "| System | Benchmark | Runs | Latest | Best | Delta | Latest Status | Budget | Seed | Scope | Lane State | Accounting | Repeatability | System State |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|",
        ]
    )

    grouped: dict[tuple[str, str], list[MatrixTrendRow]] = defaultdict(list)
    for row in trend_rows:
        grouped[(row.system, row.benchmark_id)].append(row)
    for (system, benchmark_id), group_rows in sorted(grouped.items()):
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
            f"| {system} | {benchmark_id} | {len(ordered)} | {latest_metric} | {best_metric} | "
            f"{_float_cell(delta_value)} | {latest.outcome_status} | {latest.budget} | {latest.seed} | {latest.matrix_scope} | "
            f"{latest.lane_operating_state} | {'ok' if latest.lane_budget_accounting_ok else 'incomplete'} | "
            f"{'ready' if latest.lane_repeatability_ready else 'not-ready'} | {latest.system_operating_state} |"
        )

    return "\n".join(lines)


def _float_cell(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{float(value):.6f}"
