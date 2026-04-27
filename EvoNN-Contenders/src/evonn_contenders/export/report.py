"""Markdown reporting for contender runs."""

from __future__ import annotations

from pathlib import Path

from evonn_contenders.storage import RunStore


def write_report(run_dir: str | Path) -> Path:
    """Write a compact markdown report."""
    run_dir = Path(run_dir)
    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    if not runs:
        store.close()
        raise ValueError(f"No stored runs in {run_dir}")
    run = runs[0]
    results = store.load_results(run["run_id"])
    contenders = store.load_contenders(run["run_id"])
    budget_meta = store.load_budget_metadata(run["run_id"])
    store.close()

    ok = sum(record["status"] == "ok" for record in results)
    failed = len(results) - ok
    lines = [
        f"# Contender Report: {run['run_name']}",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Seed: `{run['seed']}`",
        f"- Benchmarks: `{len(results)}`",
        f"- Contender evals: `{budget_meta.get('evaluation_count', len(contenders))}`",
        f"- Optional contenders skipped: `{budget_meta.get('optional_missing_count', 0)}`",
        f"- Successful benchmarks: `{ok}`",
        f"- Failed benchmarks: `{failed}`",
        "",
    ]
    optional_missing = budget_meta.get("optional_missing_by_group") or {}
    if optional_missing:
        lines.extend(
            [
                "## Optional Contenders Skipped",
                "",
                "| Group | Contenders |",
                "|---|---|",
            ]
        )
        for group, names in sorted(optional_missing.items()):
            lines.append(f"| {group} | {', '.join(sorted(names))} |")
        lines.append("")

    lines.extend(
        [
            "## Best Per Benchmark",
            "",
            "| Benchmark | Contender | Metric | Value | Status |",
            "|---|---|---|---:|---|",
        ]
    )
    for record in results:
        value = record["metric_value"]
        display = "---" if value is None else f"{value:.6f}"
        lines.append(
            f"| {record['benchmark_name']} | {record['contender_name']} | {record['metric_name']} | {display} | {record['status']} |"
        )

    output_path = run_dir / "report.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
