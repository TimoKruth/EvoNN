"""CLI for importing historical baseline compare artifacts into a workspace."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_compare.cli.workspace_report import refresh_workspace_reports
from evonn_compare.orchestration.historical_baseline import register_historical_baseline


def historical_baseline(
    workspace: str = typer.Argument(..., help="Active fair-matrix workspace root"),
    baseline_inputs: list[str] = typer.Argument(..., help="One or more historical baseline summary files or directories"),
    label: str | None = typer.Option(None, "--label", help="Optional cohort label for the imported baseline"),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open the refreshed dashboard in the default browser"),
) -> None:
    """Register historical baseline artifacts and rebuild the workspace decision surface."""

    workspace_path = Path(workspace)
    registration = register_historical_baseline(
        workspace=workspace_path,
        baseline_inputs=[Path(value) for value in baseline_inputs],
        label=label,
    )
    refreshed = refresh_workspace_reports(workspace=workspace_path, open_browser=open_browser)
    typer.echo(f"baseline_label\t{registration['baseline_label']}")
    typer.echo(f"baseline_root\t{registration['baseline_root']}")
    typer.echo(f"baseline_manifest\t{registration['baseline_manifest']}")
    typer.echo(f"baseline_trend_dataset\t{registration['trend_dataset']}")
    typer.echo(f"imported_summaries\t{registration['summary_count']}")
    typer.echo(f"imported_trend_rows\t{registration['trend_row_count']}")
    typer.echo(f"trend_report\t{refreshed['trend_report']}")
    typer.echo(f"dashboard\t{refreshed['dashboard']}")
    typer.echo(f"dashboard_data\t{refreshed['dashboard_data']}")
