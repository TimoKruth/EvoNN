"""Markdown renderer for canonical performance review payloads."""

from __future__ import annotations

from typing import Any


def render_performance_report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Performance Review Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline label: `{payload['baseline_label']}`",
        f"- Primary compare label: `{payload['compare_label'] or 'none'}`",
    ]
    review_references = payload.get("review_references") or {}
    if review_references:
        lines.extend(
            [
                "",
                "## Review References",
                "",
                f"- Workflow doc: `{review_references.get('workflow_doc', 'unknown')}`",
                f"- PR template: `{review_references.get('pull_request_template', 'unknown')}`",
                f"- Child issue template: `{review_references.get('child_issue_template', 'unknown')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Dataset Coverage",
            "",
            "| Label | Outcome | Rows | Accounting | Statuses | Source |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for dataset in payload["datasets"]:
        status_counts = ", ".join(f"{key}={value}" for key, value in dataset["status_counts"].items()) or "none"
        accounting_tags = ", ".join(dataset.get("accounting_tags") or []) or "full_budget"
        lines.append(
            f"| {dataset['label']} | {dataset['outcome']} | {dataset['row_count']} | {accounting_tags} | {status_counts} | {dataset['source_path']} |"
        )

    lines.extend(
        [
            "",
            "## Grouped Runtime Slices",
            "",
            "| Dataset | System | Backend | Pack | Budget | Cache | Accounting | Median Wall (s) | Median Eval/s | Failures | Quality Delta | Trust Regressions |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for dataset in payload["datasets"]:
        for row in dataset["slices"]:
            accounting_tags = ", ".join(row.get("accounting_tags") or []) or "full_budget"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(dataset["label"]),
                        str(row["system"]),
                        f"{row['backend_label']} ({row['backend_class']})",
                        str(row["pack_name"]),
                        str(row["budget"]),
                        str(row["cache_mode"]),
                        accounting_tags,
                        _float_cell(row["median_wall_clock_seconds"]),
                        _float_cell(row["median_evals_per_second"]),
                        str(row["total_failure_count"]),
                        _float_cell(row["median_quality_delta_vs_baseline"]),
                        str(row["trust_regression_count"]),
                    ]
                )
                + " |"
            )

    comparison = payload["primary_comparison"]
    lines.extend(
        [
            "",
            "## Before/After Delta View",
            "",
        ]
    )
    if not comparison["available"]:
        lines.append("_No candidate dataset selected for delta comparison._")
    else:
        summary = comparison["summary"]
        lines.extend(
            [
                f"- Baseline: `{comparison['baseline_label']}`",
                f"- Candidate: `{comparison['candidate_label']}`",
                f"- Baseline accounting: `{', '.join(summary.get('baseline_accounting_tags') or ['full_budget'])}`",
                f"- Candidate accounting: `{', '.join(summary.get('candidate_accounting_tags') or ['full_budget'])}`",
                f"- Verdict: `{summary['verdict']}`",
                f"- Median wall-clock delta: `{_float_cell(summary['median_wall_clock_delta_pct'])}%`",
                f"- Median eval/sec delta: `{_float_cell(summary['median_evals_per_second_delta_pct'])}%`",
                f"- Candidate deduplicated slots: `{summary.get('candidate_deduplicated_evaluations') or 0}`",
                f"- Candidate proxy-screened slots: `{summary.get('candidate_screened_evaluations') or 0}`",
                f"- Candidate reduced-fidelity slots: `{summary.get('candidate_reduced_fidelity_evaluations') or 0}`",
                f"- Quality regressions: `{summary['quality_regression_count']}`",
                f"- Trust regressions: `{summary['trust_regression_count']}`",
                "",
                "| System | Backend | Pack | Budget | Cache | Candidate Accounting | Matched Cases | Wall Delta % | Eval/s Delta % | Failure Delta | Quality Delta | Verdict |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in comparison["deltas"]:
            candidate_accounting = ", ".join(row.get("candidate_accounting_tags") or []) or "full_budget"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["system"]),
                        f"{row['backend_label']} ({row['backend_class']})",
                        str(row["pack_name"]),
                        str(row["budget"]),
                        str(row["cache_mode"]),
                        candidate_accounting,
                        str(row["matched_case_count"]),
                        _float_cell(row["wall_clock_delta_pct"]),
                        _float_cell(row["evals_per_second_delta_pct"]),
                        str(row["failure_count_delta"]),
                        _float_cell(row["median_quality_delta_vs_baseline"]),
                        str(row["verdict"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Optimization History",
            "",
            "| Label | Outcome | Accounting | Verdict | Matched Cases | Wall Delta % | Eval/s Delta % | Quality Regressions | Trust Regressions | Failure Delta |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["optimization_history"]:
        candidate_accounting = ", ".join(row.get("candidate_accounting_tags") or []) or "full_budget"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["outcome"]),
                    candidate_accounting,
                    str(row["verdict"]),
                    str(row["matched_case_count"]),
                    _float_cell(row["median_wall_clock_delta_pct"]),
                    _float_cell(row["median_evals_per_second_delta_pct"]),
                    str(row["quality_regression_count"]),
                    str(row["trust_regression_count"]),
                    str(row["failure_count_delta"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _float_cell(value: float | None) -> str:
    return "---" if value is None else f"{float(value):.2f}"
