"""CLI for durable promoted evidence registries."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_compare.orchestration.evidence_registry import (
    build_evidence_report,
    promote_evidence,
    validate_registry,
    validate_registry_artifacts,
)

evidence_app = typer.Typer(help="Promote and analyze decision-grade comparison evidence")


@evidence_app.command("promote")
def promote(
    inputs: list[str] = typer.Argument(..., help="Fair-matrix summary files or directories to promote"),
    registry: str = typer.Option("evidence", "--registry", help="Evidence registry directory"),
    label: str | None = typer.Option(None, "--label", help="Stable cohort label for the promoted evidence"),
    min_seeds: int = typer.Option(2, "--min-seeds", min=1, help="Minimum seeds required for decision labels"),
    copy_artifacts: bool = typer.Option(True, "--copy-artifacts/--no-copy-artifacts", help="Copy summaries into the registry"),
) -> None:
    """Promote fair-matrix summaries into a durable evidence registry."""

    result = promote_evidence(
        inputs=[Path(value) for value in inputs],
        registry=Path(registry),
        label=label,
        min_seeds=min_seeds,
        copy_artifacts=copy_artifacts,
    )
    typer.echo(f"promoted\t{result['promoted_count']}")
    typer.echo(f"registry_count\t{result['registry_count']}")
    typer.echo(f"registry\t{result['registry']}")
    typer.echo(f"index\t{result['index']}")
    typer.echo(f"report\t{result['report_json']}")
    typer.echo(f"markdown\t{result['report_md']}")


@evidence_app.command("report")
def report(
    registry: str = typer.Option("evidence", "--registry", help="Evidence registry directory"),
    min_seeds: int = typer.Option(2, "--min-seeds", min=1, help="Minimum seeds required for decision labels"),
) -> None:
    """Refresh the registry evidence report."""

    payload = build_evidence_report(registry=Path(registry), min_seeds=min_seeds)
    typer.echo(f"records\t{payload['record_count']}")
    typer.echo(f"groups\t{len(payload['groups'])}")
    typer.echo(f"report\t{Path(registry).resolve() / 'evidence_report.json'}")
    typer.echo(f"markdown\t{Path(registry).resolve() / 'evidence_report.md'}")


@evidence_app.command("validate")
def validate(
    registry: str = typer.Option("evidence", "--registry", help="Evidence registry directory"),
    require_artifacts: bool = typer.Option(False, "--require-artifacts/--no-require-artifacts", help="Fail when no registered summary artifact is readable"),
) -> None:
    """Validate evidence registry row shape."""

    result = validate_registry_artifacts(registry=Path(registry)) if require_artifacts else validate_registry(registry=Path(registry))
    typer.echo(f"ok\t{result['ok']}")
    typer.echo(f"records\t{result['record_count']}")
    for warning in result.get("warnings", []):
        typer.echo(f"warning\t{warning}")
    for issue in result["issues"]:
        typer.echo(f"issue\t{issue}")
    if not result["ok"]:
        raise typer.Exit(code=1)
