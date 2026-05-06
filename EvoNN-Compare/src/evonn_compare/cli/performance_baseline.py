"""CLI for building performance baseline bundles from compare-grade runs."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_compare.orchestration.performance_baseline import build_performance_baseline


def performance_baseline(
    inputs: list[str] = typer.Argument(..., help="Run directories or workspaces containing compare-grade exports"),
    output_root: str | None = typer.Option(None, "--output-root", help="Root directory for baseline bundles"),
    label: str | None = typer.Option(None, "--label", help="Optional baseline label"),
    write_run_artifacts: bool = typer.Option(
        True,
        "--write-run-artifacts/--no-write-run-artifacts",
        help="Refresh runtime.json, performance.json, diagnostics.json, and output_quality_report.* before aggregation",
    ),
) -> None:
    """Build a multi-budget performance baseline bundle with quality/fairness gates."""

    result = build_performance_baseline(
        inputs=[Path(value) for value in inputs],
        output_root=Path(output_root) if output_root is not None else None,
        label=label,
        write_run_artifacts=write_run_artifacts,
    )
    typer.echo(f"bundle_root\t{result['bundle_root']}")
    typer.echo(f"report\t{result['markdown']}")
    typer.echo(f"report_json\t{result['json']}")
    typer.echo(f"run_records\t{result['jsonl']}")
    typer.echo(f"runs\t{result['run_count']}")
    typer.echo(f"accepted_runs\t{result['accepted_run_count']}")
    typer.echo(f"systems\t{result['system_count']}")
