"""CLI for refreshing the fair-matrix workspace decision surface."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import webbrowser

import typer

from evonn_compare.cli.trend_report import load_trend_rows
from evonn_compare.reporting.fair_matrix_dashboard import (
    build_dashboard_payload,
    discover_fair_matrix_summaries,
    render_dashboard_html,
)
from evonn_compare.reporting.fair_matrix_trends_md import render_fair_matrix_trend_markdown


def workspace_report(
    workspace: str = typer.Argument(..., help="Fair-matrix workspace root"),
    trend_output: str | None = typer.Option(None, "--trend-output", help="Workspace trend markdown output path"),
    dashboard_output: str | None = typer.Option(None, "--dashboard-output", help="Workspace dashboard HTML output path"),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open the refreshed dashboard in the default browser"),
) -> None:
    """Refresh workspace trend and dashboard artifacts from canonical JSON outputs."""

    artifact_paths = refresh_workspace_reports(
        workspace=Path(workspace),
        trend_output=Path(trend_output) if trend_output is not None else None,
        dashboard_output=Path(dashboard_output) if dashboard_output is not None else None,
        open_browser=open_browser,
    )
    typer.echo(f"workspace\t{artifact_paths['workspace']}")
    typer.echo(f"summary-count\t{artifact_paths['summary_count']}")
    typer.echo(f"trend-dataset\t{artifact_paths['trend_dataset']}")
    typer.echo(f"trend-report\t{artifact_paths['trend_report']}")
    typer.echo(f"trend-report-data\t{artifact_paths['trend_report_data']}")
    typer.echo(f"dashboard\t{artifact_paths['dashboard']}")
    typer.echo(f"dashboard-data\t{artifact_paths['dashboard_data']}")
    if open_browser:
        typer.echo(f"opened\t{Path(artifact_paths['dashboard']).resolve().as_uri()}")


def refresh_workspace_reports(
    *,
    workspace: Path,
    trend_output: Path | None = None,
    dashboard_output: Path | None = None,
    open_browser: bool = False,
) -> dict[str, str | int]:
    workspace_path = workspace.resolve()
    trend_dataset_path = workspace_path / "trends" / "fair_matrix_trend_rows.jsonl"
    trend_report_path = trend_output or (workspace_path / "trends" / "fair_matrix_trends.md")
    trend_report_data_path = trend_report_path.with_suffix(".json")
    dashboard_output_path = dashboard_output or (workspace_path / "fair_matrix_dashboard.html")
    summary_paths = discover_fair_matrix_summaries([workspace_path / "reports", workspace_path])

    if not trend_dataset_path.exists() and not summary_paths:
        raise typer.BadParameter(
            "workspace does not contain fair-matrix trend rows or fair_matrix_summary.json artifacts"
        )

    trend_report_path.parent.mkdir(parents=True, exist_ok=True)
    if trend_dataset_path.exists():
        trend_rows = load_trend_rows([trend_dataset_path])
        trend_report_path.write_text(render_fair_matrix_trend_markdown(trend_rows), encoding="utf-8")
        trend_report_data_path.write_text(
            json.dumps([asdict(row) for row in trend_rows], indent=2, default=str),
            encoding="utf-8",
        )
    else:
        trend_report_path.write_text("# Fair Matrix Trends: empty\n\n_No trend rows found._", encoding="utf-8")
        trend_report_data_path.write_text("[]\n", encoding="utf-8")

    dashboard_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload(summary_paths, output_path=dashboard_output_path) if summary_paths else {
        "generated_at": None,
        "summary_count": 0,
        "packs": [],
        "budgets": [],
        "lane_counts": {
            "fair": 0,
            "contract_fair": 0,
            "trusted_core": 0,
            "trusted_extended": 0,
            "repeatable": 0,
            "artifact_complete": 0,
        },
        "runs": [],
        "leaderboards": {"all_systems": [], "projects_only": []},
    }
    dashboard_output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
    dashboard_output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if open_browser:
        webbrowser.open(dashboard_output_path.resolve().as_uri())

    return {
        "workspace": str(workspace_path),
        "summary_count": len(summary_paths),
        "trend_dataset": str(trend_dataset_path),
        "trend_report": str(trend_report_path),
        "trend_report_data": str(trend_report_data_path),
        "dashboard": str(dashboard_output_path),
        "dashboard_data": str(dashboard_output_path.with_suffix('.json')),
    }
