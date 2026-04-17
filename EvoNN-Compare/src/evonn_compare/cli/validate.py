"""CLI validate command."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor


def validate(
    run_dir: str = typer.Argument(..., help="Exported run directory"),
    pack: str = typer.Option(..., "--pack", help="Parity pack name or YAML path"),
) -> None:
    """Validate one exported run against a parity pack."""

    ingestor = SystemIngestor(Path(run_dir))
    report = ingestor.validate(load_parity_pack(pack))
    typer.echo(f"ok\t{str(report.ok).lower()}")
    for issue in report.issues:
        typer.echo(f"{issue.level}\t{issue.code}\t{issue.message}")
