"""CLI for fair four-way matrix campaigns."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evonn_compare.contracts.parity import resolve_pack_path
from evonn_compare.orchestration.fair_matrix import (
    prepare_fair_matrix_cases,
    run_fair_matrix_case,
)


def fair_matrix(
    pack: str = typer.Option(..., "--pack", help="Parity pack name or YAML path"),
    seeds: str = typer.Option("42", "--seeds", help="Comma-separated seeds"),
    budgets: str = typer.Option("76", "--budgets", help="Comma-separated budgets"),
    workspace: str = typer.Option(..., "--workspace", help="Campaign workspace"),
    prism_root: str = typer.Option("EvoNN-Prism", "--prism-root"),
    topograph_root: str = typer.Option("EvoNN-Topograph", "--topograph-root"),
    stratograph_root: str = typer.Option("EvoNN-Stratograph", "--stratograph-root"),
    contenders_root: str = typer.Option("EvoNN-Contenders", "--contenders-root"),
    parallel: bool = typer.Option(True, "--parallel/--serial", help="Run project stages concurrently"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only generate configs and print cases"),
) -> None:
    """Generate and optionally execute fair four-way compare cases."""

    pack_path = resolve_pack_path(pack)
    paths, cases = prepare_fair_matrix_cases(
        pack_name=Path(pack_path).stem,
        base_pack_path=pack_path,
        seeds=_parse_csv_ints(seeds),
        budgets=_parse_csv_ints(budgets),
        workspace=Path(workspace),
        prism_root=Path(prism_root),
        topograph_root=Path(topograph_root),
        stratograph_root=Path(stratograph_root),
        contenders_root=Path(contenders_root),
    )
    if dry_run:
        typer.echo("mode\tdry-run")
        typer.echo(str(paths.manifest_path))
        for case in cases:
            typer.echo(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in case.__dict__.items()}))
        return

    typer.echo("mode\texecute")
    for case in cases:
        summary_path = run_fair_matrix_case(
            case,
            prism_root=Path(prism_root),
            topograph_root=Path(topograph_root),
            stratograph_root=Path(stratograph_root),
            contenders_root=Path(contenders_root),
            parallel=parallel,
        )
        typer.echo(f"summary\t{summary_path}")


def _parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return [int(item) for item in values]
