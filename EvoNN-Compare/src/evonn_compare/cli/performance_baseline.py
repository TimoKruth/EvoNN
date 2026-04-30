"""CLI for creating canonical planned performance-baseline artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess

import typer

from evonn_compare.orchestration.performance_baseline import initialize_performance_baseline


def performance_baseline(
    output_root: str | None = typer.Option(
        None,
        "--output-root",
        help="Baseline output root; defaults to EvoNN-Compare/performance_baselines/<date>-<git-sha>",
    ),
    packs: str = typer.Option("tier1_core,tier_b_core", "--packs", help="Comma-separated parity packs"),
    budgets: str = typer.Option("64,256,1000", "--budgets", help="Comma-separated evaluation budgets"),
    seeds: str = typer.Option("42,43,44", "--seeds", help="Comma-separated seeds"),
    cache_modes: str = typer.Option("cold,warm", "--cache-modes", help="Comma-separated cache modes"),
    systems: str = typer.Option(
        "prism,topograph,stratograph,primordia,contenders",
        "--systems",
        help="Comma-separated system ids to include in the matrix",
    ),
    matrix_workspaces: list[str] = typer.Option(
        [],
        "--matrix-workspace",
        help="Repeatable fair-matrix workspace(s) to import measured run artifacts from",
    ),
) -> None:
    """Create the canonical planned performance-baseline artifact bundle."""

    result = initialize_performance_baseline(
        output_root=_resolve_output_root(output_root),
        packs=_parse_csv_strings(packs),
        budgets=_parse_csv_ints(budgets),
        seeds=_parse_csv_ints(seeds),
        cache_modes=_parse_csv_strings(cache_modes),
        systems=_parse_csv_strings(systems),
        matrix_workspaces=[Path(value) for value in matrix_workspaces],
    )
    typer.echo(f"mode\t{result['mode']}")
    typer.echo(f"baseline_root\t{result['baseline_root']}")
    typer.echo(f"baseline_manifest\t{result['baseline_manifest']}")
    typer.echo(f"baseline_summary\t{result['baseline_summary']}")
    typer.echo(f"raw_runs\t{result['raw_runs']}")
    typer.echo(f"perf_rows\t{result['perf_rows']}")
    typer.echo(f"perf_dashboard\t{result['perf_dashboard']}")
    typer.echo(f"perf_dashboard_json\t{result['perf_dashboard_json']}")
    typer.echo(f"planned_case_count\t{result['planned_case_count']}")


def _resolve_output_root(raw: str | None) -> Path:
    if raw is not None:
        return Path(raw)
    compare_root = Path(__file__).resolve().parents[3]
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    git_sha = _resolve_git_sha(compare_root.parent)
    return compare_root / "performance_baselines" / f"{date_prefix}-{git_sha}"


def _parse_csv_strings(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise typer.BadParameter("expected at least one value")
    return values


def _parse_csv_ints(raw: str) -> list[int]:
    try:
        return [int(value) for value in _parse_csv_strings(raw)]
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated integers") from exc


def _resolve_git_sha(workspace_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
