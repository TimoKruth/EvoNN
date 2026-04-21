"""Reporting helpers for Primordia exports."""
from __future__ import annotations

from pathlib import Path
import json


def write_report(run_dir: str | Path) -> Path:
    """Return the existing Primordia report path, regenerating from run summary if missing."""

    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"
    if report_path.exists():
        return report_path

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines = [
            "# Primordia Run Report",
            "",
            f"- Run ID: `{summary.get('run_id', run_dir.name)}`",
            f"- Runtime: `{summary.get('runtime', 'unknown')}`",
            f"- Runtime Version: `{summary.get('runtime_version') or 'unknown'}`",
            f"- Evaluations: `{summary.get('evaluation_count', 0)}`",
            f"- Benchmarks: `{summary.get('benchmark_count', 0)}`",
            f"- Budget Policy: `{summary.get('budget_policy_name', 'unknown')}`",
            f"- Wall Clock Seconds: `{float(summary.get('wall_clock_seconds', 0.0)):.3f}`",
            "",
            "## Primitive Usage",
            "",
            "| Primitive Family | Evaluations |",
            "|---|---:|",
        ]
        primitive_usage = summary.get("primitive_usage", {})
        if primitive_usage:
            for family, count in primitive_usage.items():
                lines.append(f"| {family} | {count} |")
        else:
            lines.append("| none | 0 |")
        lines.extend([
            "",
            "## Benchmark Group Coverage",
            "",
            "| Group | Benchmarks |",
            "|---|---:|",
        ])
        group_counts = summary.get("group_counts", {})
        if group_counts:
            for group, count in group_counts.items():
                lines.append(f"| {group} | {count} |")
        else:
            lines.append("| none | 0 |")
        lines.extend([
            "",
            "## Failure Summary",
            "",
            f"- Failure Count: `{int(summary.get('failure_count', 0))}`",
            "",
            "## Best Primitive Per Benchmark",
            "",
            "| Benchmark | Primitive | Metric | Value | Status |",
            "|---|---|---|---:|---|",
        ])
        for record in summary.get("best_results", []):
            value = "---" if record.get("metric_value") is None else f"{float(record['metric_value']):.6f}"
            lines.append(
                f"| {record['benchmark_name']} | {record['primitive_name']} | {record['metric_name']} | {value} | {record['status']} |"
            )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    best_path = run_dir / "best_results.json"
    records = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else []
    lines = [
        "# Primordia Export Report",
        "",
        "| Benchmark | Primitive | Metric | Value | Status |",
        "|---|---|---|---:|---|",
    ]
    for record in records:
        value = "---" if record.get("metric_value") is None else f"{float(record['metric_value']):.6f}"
        lines.append(
            f"| {record['benchmark_name']} | {record['primitive_name']} | {record['metric_name']} | {value} | {record['status']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
