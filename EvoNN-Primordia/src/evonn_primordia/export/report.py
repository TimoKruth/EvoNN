"""Reporting helpers for Primordia exports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_best_results(run_dir: str | Path, summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Load benchmark winners from the summary first, then fall back to best_results.json."""

    summary_best_results = list((summary or {}).get("best_results") or [])
    if summary_best_results:
        return summary_best_results

    best_results_path = Path(run_dir) / "best_results.json"
    if not best_results_path.exists():
        return []

    try:
        loaded = json.loads(best_results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    return loaded if isinstance(loaded, list) else []


def build_primitive_bank_summary(
    *,
    summary: dict[str, Any],
    best_results: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in trial_records:
        family = str(record.get("primitive_family") or "unknown")
        by_family.setdefault(family, []).append(record)

    best_by_family: dict[str, dict[str, Any]] = {}
    for family, records in by_family.items():
        ok_records = [record for record in records if record.get("status") == "ok"]
        source = ok_records or records
        best_by_family[family] = max(
            source,
            key=lambda record: float(record.get("quality")) if record.get("quality") is not None else float("-inf"),
        )

    wins: dict[str, list[str]] = {}
    for record in best_results:
        if record.get("status") != "ok":
            continue
        family = str(record.get("primitive_family") or "unknown")
        benchmark_name = str(record.get("benchmark_name") or "")
        if benchmark_name:
            wins.setdefault(family, []).append(benchmark_name)

    usage = summary.get("primitive_usage", {})
    families = sorted(set(usage) | set(best_by_family) | set(wins))
    primitive_families = []
    for family in families:
        representative = best_by_family.get(family, {})
        won = sorted(wins.get(family, []))
        primitive_families.append(
            {
                "family": family,
                "evaluation_count": int(usage.get(family, 0)),
                "benchmark_wins": len(won),
                "benchmarks_won": won,
                "best_metric_name": representative.get("metric_name"),
                "best_metric_value": representative.get("metric_value"),
                "representative_genome_id": representative.get("genome_id"),
                "representative_architecture_summary": representative.get("architecture_summary"),
            }
        )

    return {
        "system": "primordia",
        "run_id": summary.get("run_id"),
        "run_name": summary.get("run_name"),
        "runtime": summary.get("runtime"),
        "runtime_version": summary.get("runtime_version"),
        "primitive_families": primitive_families,
    }



def write_report(run_dir: str | Path) -> Path:
    """Return the existing Primordia report path, regenerating from run summary if missing."""

    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"
    if report_path.exists():
        return report_path

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trial_records_path = run_dir / "trial_records.json"
        primitive_bank_path = run_dir / "primitive_bank_summary.json"
        best_results = load_best_results(run_dir, summary)
        trial_records = json.loads(trial_records_path.read_text(encoding="utf-8")) if trial_records_path.exists() else []
        primitive_bank = (
            json.loads(primitive_bank_path.read_text(encoding="utf-8"))
            if primitive_bank_path.exists()
            else build_primitive_bank_summary(summary=summary, best_results=best_results, trial_records=trial_records)
        )

        lines = [
            "# Primordia Run Report",
            "",
            f"- Run ID: `{summary.get('run_id', run_dir.name)}`",
            f"- Runtime: `{summary.get('runtime', 'unknown')}`",
            f"- Runtime Version: `{summary.get('runtime_version') or 'unknown'}`",
            f"- Evaluations: `{summary.get('evaluation_count', 0)}`",
            f"- Target Evaluations: `{summary.get('target_evaluation_count', 'n/a')}`",
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
            "## Primitive Bank Summary",
            "",
            "| Family | Evaluations | Benchmark Wins | Won Benchmarks | Best Metric | Best Value | Representative Genome | Representative Architecture |",
            "|---|---:|---:|---|---|---:|---|---|",
        ])
        primitive_rows = primitive_bank.get("primitive_families") or []
        if primitive_rows:
            for row in primitive_rows:
                won = ", ".join(row.get("benchmarks_won") or []) or "—"
                best_value = row.get("best_metric_value")
                rendered_value = "---" if best_value is None else f"{float(best_value):.6f}"
                lines.append(
                    "| {family} | {evaluation_count} | {benchmark_wins} | {won} | {best_metric} | {best_value} | {genome} | {architecture} |".format(
                        family=row.get("family", "unknown"),
                        evaluation_count=int(row.get("evaluation_count", 0)),
                        benchmark_wins=int(row.get("benchmark_wins", 0)),
                        won=won,
                        best_metric=row.get("best_metric_name") or "—",
                        best_value=rendered_value,
                        genome=row.get("representative_genome_id") or "—",
                        architecture=row.get("representative_architecture_summary") or "—",
                    )
                )
        else:
            lines.append("| none | 0 | 0 | — | — | --- | — | — |")
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
        for record in best_results:
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
