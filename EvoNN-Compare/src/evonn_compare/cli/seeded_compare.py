"""CLI for canonical seeded-vs-unseeded compare publication."""

from __future__ import annotations

from pathlib import Path
import webbrowser

import typer

from evonn_compare.orchestration.seeded_compare import publish_seeded_vs_unseeded_workspace


def seeded_compare(
    workspace: str = typer.Option(..., "--workspace", help="Workspace root for canonical seeded-vs-unseeded artifacts"),
    pack: str = typer.Option("tier1_core_smoke", "--pack", help="Base parity pack name or YAML path"),
    seed: int = typer.Option(42, "--seed", help="Campaign seed"),
    budget: int | None = typer.Option(None, "--budget", help="Evaluation budget override; defaults to the pack budget"),
    primordia_root: str = typer.Option("EvoNN-Primordia", "--primordia-root"),
    topograph_root: str = typer.Option("EvoNN-Topograph", "--topograph-root"),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open the refreshed dashboard in the default browser"),
) -> None:
    """Publish the canonical Primordia->Topograph seeded-vs-unseeded workspace."""

    artifacts = publish_seeded_vs_unseeded_workspace(
        workspace=Path(workspace),
        pack_name=pack,
        seed=seed,
        budget=budget,
        primordia_root=Path(primordia_root),
        topograph_root=Path(topograph_root),
    )
    for key in (
        "workspace",
        "pack_path",
        "primordia_run_dir",
        "seed_artifact",
        "unseeded_run_dir",
        "seeded_run_dir",
        "seeded_vs_unseeded_report",
        "seeded_vs_unseeded_data",
        "trend_dataset",
        "trend_report",
        "trend_report_data",
        "dashboard",
        "dashboard_data",
    ):
        typer.echo(f"{key}\t{artifacts[key]}")
    if open_browser:
        dashboard_url = Path(str(artifacts["dashboard"])).resolve().as_uri()
        webbrowser.open(dashboard_url)
        typer.echo(f"opened\t{dashboard_url}")
