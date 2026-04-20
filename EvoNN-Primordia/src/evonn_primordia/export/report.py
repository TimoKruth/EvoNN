"""Reporting helpers for Primordia exports."""
from __future__ import annotations

from pathlib import Path
import json


def write_report(run_dir: str | Path) -> Path:
    """Return the existing Primordia report path, regenerating from best results if missing."""

    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"
    if report_path.exists():
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
