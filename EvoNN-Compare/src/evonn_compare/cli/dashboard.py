"""CLI for building a static fair-matrix dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import webbrowser

import typer

from evonn_compare.reporting.fair_matrix_dashboard import (
    build_dashboard_payload,
    discover_fair_matrix_summaries,
    render_dashboard_html,
)


def dashboard(
    inputs: list[str] | None = typer.Argument(
        None,
        help="One or more fair-matrix summary files or directories. Defaults to EvoNN-Compare/manual_compare_runs.",
    ),
    output: str = typer.Option(
        "EvoNN-Compare/manual_compare_runs/fair_matrix_dashboard.html",
        "--output",
        help="HTML output path",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the rendered dashboard in the default browser",
    ),
) -> None:
    """Build a static HTML dashboard for accumulated fair-matrix runs."""

    output_path = Path(output)
    summary_paths = discover_fair_matrix_summaries([Path(value) for value in inputs] if inputs else None)
    if not summary_paths:
        raise typer.BadParameter("no fair_matrix_summary.json files found in the supplied inputs")

    payload = build_dashboard_payload(summary_paths, output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dashboard_url = output_path.resolve().as_uri()
    if open_browser:
        webbrowser.open(dashboard_url)

    typer.echo(f"summaries\t{len(summary_paths)}")
    typer.echo(f"dashboard\t{output_path.resolve()}")
    typer.echo(f"data\t{output_path.with_suffix('.json').resolve()}")
    if open_browser:
        typer.echo(f"opened\t{dashboard_url}")
