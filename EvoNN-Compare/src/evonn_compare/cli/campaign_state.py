"""CLI commands for inspecting and controlling research campaign workspace state."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evonn_compare.orchestration.campaign_state import (
    load_workspace_state,
    render_workspace_state_markdown,
    request_stop,
    summarize_workspace_state,
    workspace_state_path,
)


def campaign_inspect(
    workspace: str = typer.Argument(..., help="Campaign or fair-matrix workspace root"),
    as_json: bool = typer.Option(False, "--json", help="Print the raw workspace state JSON"),
) -> None:
    """Inspect mutable campaign state without rerunning engines."""

    workspace_path = Path(workspace)
    state = load_workspace_state(workspace_path)
    if state is None:
        raise typer.BadParameter(f"workspace state not found: {workspace_state_path(workspace_path)}")
    if as_json:
        typer.echo(json.dumps(state, indent=2))
        return
    summary = summarize_workspace_state(state)
    typer.echo(f"workspace\t{workspace_path.resolve()}")
    typer.echo(f"state\t{workspace_state_path(workspace_path)}")
    typer.echo(f"kind\t{state.get('workspace_kind', 'unknown')}")
    typer.echo(f"manifest\t{state.get('manifest_path', 'unknown')}")
    typer.echo(f"cases\t{summary['case_count']}")
    for status, count in summary["status_counts"].items():
        typer.echo(f"status_{status}\t{count}")
    typer.echo(f"resumed_cases\t{summary['resumed_case_count']}")
    typer.echo(f"integrity_failed_cases\t{summary['integrity_failed_count']}")
    typer.echo(f"stop_requested\t{summary['stop_requested']}")
    typer.echo("")
    typer.echo(render_workspace_state_markdown(state))


def campaign_stop(
    workspace: str = typer.Argument(..., help="Campaign or fair-matrix workspace root"),
) -> None:
    """Request a graceful stop at the next orchestration stage boundary."""

    workspace_path = Path(workspace)
    state = request_stop(workspace_path)
    typer.echo(f"workspace\t{workspace_path.resolve()}")
    typer.echo(f"state\t{workspace_state_path(workspace_path)}")
    typer.echo(f"kind\t{state.get('workspace_kind', 'unknown')}")
    typer.echo("stop_requested\ttrue")
