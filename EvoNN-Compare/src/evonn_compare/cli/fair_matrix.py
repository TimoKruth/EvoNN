"""CLI for fair four-way matrix campaigns."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evonn_compare.contracts.parity import resolve_pack_path
from evonn_compare.orchestration.lane_presets import lane_preset_help, resolve_lane_preset
from evonn_compare.orchestration.fair_matrix import (
    prepare_fair_matrix_cases,
    run_fair_matrix_case,
)


def fair_matrix(
    pack: str | None = typer.Option(None, "--pack", help="Parity pack name or YAML path"),
    preset: str | None = typer.Option(None, "--preset", help=lane_preset_help(default_name="smoke")),
    seeds: str | None = typer.Option(None, "--seeds", help="Comma-separated seeds"),
    budgets: str | None = typer.Option(None, "--budgets", help="Comma-separated budgets"),
    workspace: str = typer.Option(..., "--workspace", help="Campaign workspace"),
    prism_root: str = typer.Option("EvoNN-Prism", "--prism-root"),
    topograph_root: str = typer.Option("EvoNN-Topograph", "--topograph-root"),
    stratograph_root: str = typer.Option("EvoNN-Stratograph", "--stratograph-root"),
    primordia_root: str = typer.Option("EvoNN-Primordia", "--primordia-root"),
    contenders_root: str = typer.Option("EvoNN-Contenders", "--contenders-root"),
    include_contenders: bool = typer.Option(True, "--include-contenders/--no-contenders", help="Include contender baselines in the fair-matrix run"),
    parallel: bool = typer.Option(True, "--parallel/--serial", help="Run project stages concurrently"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only generate configs and print cases"),
) -> None:
    """Generate and optionally execute fair four-way compare cases."""

    preset_name = preset or (None if pack else "smoke")
    preset_spec = resolve_lane_preset(preset_name) if preset_name else None
    pack_name = pack or (preset_spec.pack if preset_spec else None)

    pack_path = resolve_pack_path(pack_name)
    paths, cases = prepare_fair_matrix_cases(
        pack_name=Path(pack_path).stem,
        base_pack_path=pack_path,
        seeds=_parse_optional_csv_ints(seeds) or (list(preset_spec.seeds) if preset_spec else [42]),
        budgets=_parse_optional_csv_ints(budgets) or (list(preset_spec.budgets) if preset_spec else [76]),
        workspace=Path(workspace),
        prism_root=Path(prism_root),
        topograph_root=Path(topograph_root),
        stratograph_root=Path(stratograph_root),
        primordia_root=Path(primordia_root),
        contenders_root=Path(contenders_root),
        include_contenders=include_contenders,
        lane_preset=preset,
    )
    if dry_run:
        typer.echo("mode\tdry-run")
        typer.echo(f"manifest\t{paths.manifest_path}")
        typer.echo(f"trend-dataset\t{paths.trends_dir / 'fair_matrix_trends.jsonl'}")
        for case in cases:
            typer.echo(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in case.__dict__.items()}))
        return

    typer.echo("mode\texecute")
    typer.echo(f"manifest\t{paths.manifest_path}")
    typer.echo(f"trend-dataset\t{paths.trends_dir / 'fair_matrix_trends.jsonl'}")
    for case in cases:
        summary_path = run_fair_matrix_case(
            case,
            prism_root=Path(prism_root),
            topograph_root=Path(topograph_root),
            stratograph_root=Path(stratograph_root),
            primordia_root=Path(primordia_root),
            contenders_root=Path(contenders_root),
            parallel=parallel,
        )
        typer.echo(f"summary\t{summary_path}")
        for label, artifact_path in _trend_artifact_paths(summary_path).items():
            typer.echo(f"{label}\t{artifact_path}")


def _trend_artifact_paths(summary_path: Path) -> dict[str, Path]:
    case_dir = summary_path.parent
    workspace_dir = case_dir.parent
    return {
        "trend_rows": case_dir / "trend_rows.json",
        "trend_report": case_dir / "trend_report.md",
        "workspace_trend_rows": workspace_dir / "fair_matrix_trend_rows.jsonl",
        "workspace_trend_report": workspace_dir / "fair_matrix_trends.md",
    }


def _parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return [int(item) for item in values]


def _parse_optional_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return _parse_csv_ints(raw)
