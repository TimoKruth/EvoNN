"""CLI for refreshing the fair-matrix workspace decision surface."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import webbrowser

import typer

from evonn_compare.cli.trend_report import load_trend_rows
from evonn_compare.orchestration.historical_baseline import discover_workspace_trend_inputs
from evonn_compare.orchestration.campaign_state import load_workspace_state, render_workspace_state_markdown
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
    campaign_state = load_workspace_state(workspace_path)
    trend_report_path = trend_output or (workspace_path / "trends" / "fair_matrix_trends.md")
    trend_report_data_path = trend_report_path.with_suffix(".json")
    dashboard_output_path = dashboard_output or (workspace_path / "fair_matrix_dashboard.html")
    summary_paths = discover_fair_matrix_summaries([workspace_path / "reports", workspace_path / "baselines", workspace_path])
    trend_inputs = discover_workspace_trend_inputs(workspace_path, summary_paths=summary_paths)
    trend_dataset_path = trend_report_path.parent / "fair_matrix_trend_rows.jsonl"

    if not trend_inputs and not summary_paths and campaign_state is None:
        raise typer.BadParameter(
            "workspace does not contain state, fair-matrix trend rows, or fair_matrix_summary.json artifacts"
        )

    trend_report_path.parent.mkdir(parents=True, exist_ok=True)
    state_markdown = render_workspace_state_markdown(campaign_state) if campaign_state is not None else None
    if trend_inputs:
        trend_rows = load_trend_rows(trend_inputs)
        trend_dataset_path.write_text(
            "".join(json.dumps(asdict(row), default=str) + "\n" for row in trend_rows),
            encoding="utf-8",
        )
        trend_body = render_fair_matrix_trend_markdown(trend_rows)
        if state_markdown is not None:
            trend_body = state_markdown + "\n\n" + trend_body
        trend_report_path.write_text(trend_body, encoding="utf-8")
        trend_report_data_path.write_text(
            json.dumps([asdict(row) for row in trend_rows], indent=2, default=str),
            encoding="utf-8",
        )
    else:
        empty_trends = "# Fair Matrix Trends: empty\n\n_No trend rows found._"
        if state_markdown is not None:
            empty_trends = state_markdown + "\n\n" + empty_trends
        trend_report_path.write_text(empty_trends, encoding="utf-8")
        trend_report_data_path.write_text("[]\n", encoding="utf-8")

    dashboard_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload(summary_paths, output_path=dashboard_output_path, campaign_state=campaign_state) if summary_paths or campaign_state is not None else {
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
        "specialization": {
            "family_leaderboards": {"all_systems": [], "projects_only": []},
            "engine_profiles": {"all_systems": [], "projects_only": []},
        },
        "multi_seed": {"all_systems": [], "projects_only": []},
        "seed_scorecards": {"all_systems": [], "projects_only": []},
        "transfer": {"case_count": 0, "cases": [], "regimes": {}, "family_rows": []},
        "campaign_state": {
            "available": False,
            "workspace_kind": None,
            "case_count": 0,
            "status_counts": {},
            "resumed_case_count": 0,
            "integrity_failed_count": 0,
            "stop_requested": False,
            "cases": [],
        },
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
