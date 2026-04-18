"""Markdown renderer for four-way fairness matrix summaries."""

from __future__ import annotations

from evonn_compare.comparison.fair_matrix import FairMatrixSummary, SYSTEM_ORDER
from evonn_compare.adapters.slots import system_display_name


def render_fair_matrix_markdown(summary: FairMatrixSummary) -> str:
    lines = [
        f"# Fair Matrix: {summary.pack_name}",
        "",
        "## Fair Search-Budget Results",
        "",
        _result_header(include_note=False),
    ]
    if summary.fair_rows:
        for row in summary.fair_rows:
            lines.append(_result_row(row, include_note=False))
    else:
        lines.append("_None_")

    lines.extend(
        [
            "",
            "## Reference Baseline Results",
            "",
            _result_header(include_note=True),
        ]
    )
    if summary.reference_rows:
        for row in summary.reference_rows:
            lines.append(_result_row(row, include_note=True))
    else:
        lines.append("_None_")

    lines.extend(
        [
            "",
            "## Parity/Validity Check",
            "",
            "| Budget | Seed | Pair | Status | Left Evals | Right Evals | Left Policy | Right Policy | Data Sig Match | Reason | Report |",
            "|---:|---:|---|---|---:|---:|---|---|---|---|---|",
        ]
    )
    if summary.parity_rows:
        for row in summary.parity_rows:
            lines.append(
                f"| {row.budget} | {row.seed} | {row.pair_label} | {row.parity_status} | "
                f"{row.left_eval_count} | {row.right_eval_count} | {row.left_policy or '---'} | "
                f"{row.right_policy or '---'} | {'yes' if row.data_signature_match else 'no'} | "
                f"{row.reason or '---'} | {row.comparison_report} |"
            )
    else:
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    return "\n".join(lines)


def _result_header(*, include_note: bool) -> str:
    columns = ["Budget", "Seed", "Benchmarks"]
    for system in SYSTEM_ORDER:
        label = system_display_name(system)
        columns.extend([f"{label} Evals", f"{label} Wins"])
    columns.append("Ties")
    if include_note:
        columns.append("Note")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---:" if column in {"Budget", "Seed", "Benchmarks", "Ties"} or column.endswith("Wins") or column.endswith("Evals") else "---" for column in columns) + " |"
    return "\n".join([header, divider])


def _result_row(row, *, include_note: bool) -> str:
    values = [str(row.budget), str(row.seed), str(row.benchmark_count)]
    for system in SYSTEM_ORDER:
        values.append(str(row.evaluation_counts.get(system, 0)))
        values.append(str(row.wins.get(system, 0)))
    values.append(str(row.ties))
    if include_note:
        values.append(row.note or "---")
    return "| " + " | ".join(values) + " |"
