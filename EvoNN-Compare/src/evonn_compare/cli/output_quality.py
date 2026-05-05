"""CLI for output-quality and performance artifact inspection."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_compare.output_quality import inspect_paths, write_aggregate_report


def output_quality(
    inputs: list[str] = typer.Argument(..., help="Run directories or fair-matrix workspaces to inspect"),
    output: str | None = typer.Option(None, "--output", help="Aggregate markdown report output path"),
    write_run_artifacts: bool = typer.Option(
        True,
        "--write-run-artifacts/--no-write-run-artifacts",
        help="Write runtime.json, performance.json, diagnostics.json, and output_quality_report.* into each run dir",
    ),
) -> None:
    """Inspect compare-grade outputs and generate normalized measurement artifacts."""

    records = inspect_paths([Path(value) for value in inputs], write_run_artifacts=write_run_artifacts)
    output_path = Path(output) if output is not None else Path(inputs[-1]) / "output_quality_overview.md"
    aggregate = write_aggregate_report(records, output_path.resolve())
    typer.echo(f"runs\t{len(records)}")
    typer.echo(f"report\t{aggregate['markdown']}")
    typer.echo(f"report_json\t{aggregate['json']}")
    for record in records:
        typer.echo(
            "run\t{system}\t{level}\t{measurement}\t{path}".format(
                system=record.system,
                level=record.quality_level,
                measurement=record.measurement_state,
                path=record.run_dir,
            )
        )
