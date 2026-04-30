"""CLI for compare-facing transfer-regime workspaces."""

from __future__ import annotations

import json
from pathlib import Path
import webbrowser

import typer

from evonn_compare.contracts.parity import load_parity_pack, resolve_pack_path
from evonn_compare.orchestration.lane_presets import lane_preset_help, resolve_lane_preset
from evonn_compare.orchestration.transfer_regimes import publish_transfer_regime_workspace


def transfer_regimes(
    pack: str | None = typer.Option(None, "--pack", help="Parity pack name or YAML path"),
    preset: str | None = typer.Option(None, "--preset", help=lane_preset_help(default_name="tier_b_local")),
    seeds: str | None = typer.Option(None, "--seeds", help="Comma-separated seeds"),
    budget: int | None = typer.Option(None, "--budget", help="Evaluation budget override; defaults to the preset or pack budget"),
    workspace: str = typer.Option(..., "--workspace", help="Workspace root for transfer-regime artifacts"),
    primordia_root: str = typer.Option("EvoNN-Primordia", "--primordia-root"),
    topograph_root: str = typer.Option("EvoNN-Topograph", "--topograph-root"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only print the resolved plan and output paths"),
    open_browser: bool = typer.Option(False, "--open/--no-open", help="Open the refreshed dashboard in the default browser"),
) -> None:
    """Publish no-seed, direct, and staged transfer regimes from one compare-facing surface."""

    preset_name = preset or (None if pack else "tier_b_local")
    preset_spec = resolve_lane_preset(preset_name) if preset_name else None
    pack_name = pack or (preset_spec.pack if preset_spec else None)
    if pack_name is None:
        raise typer.BadParameter("either --pack or --preset must resolve to a parity pack")

    seed_values = _parse_optional_csv_ints(seeds) or (list(preset_spec.seeds) if preset_spec else [42])
    pack_path = resolve_pack_path(pack_name)
    pack_spec = load_parity_pack(pack_path)
    effective_budget = budget or (preset_spec.budgets[0] if preset_spec else pack_spec.budget_policy.evaluation_count)
    workspace_path = Path(workspace).resolve()

    if dry_run:
        typer.echo("mode\tdry-run")
        typer.echo(f"workspace\t{workspace_path}")
        typer.echo(f"pack_path\t{pack_path}")
        typer.echo(f"budget\t{effective_budget}")
        typer.echo(f"trend_dataset\t{workspace_path / 'trends' / 'fair_matrix_trend_rows.jsonl'}")
        typer.echo(f"transfer_summary_report\t{workspace_path / 'reports' / 'transfer_regime_summary.md'}")
        typer.echo(f"transfer_summary_data\t{workspace_path / 'reports' / 'transfer_regime_summary.json'}")
        for seed in seed_values:
            typer.echo(
                json.dumps(
                    {
                        "seed": seed,
                        "primordia_run_dir": str(workspace_path / "runs" / "primordia" / f"{Path(pack_path).stem}_seed{seed}_source"),
                        "direct_seed_artifact": str(workspace_path / "seed_artifacts" / f"seed{seed}_direct_quality.json"),
                        "staged_seed_artifact": str(workspace_path / "seed_artifacts" / f"seed{seed}_staged_seed_candidates.json"),
                        "regimes": ["none", "direct", "staged"],
                    }
                )
            )
        return

    artifacts = publish_transfer_regime_workspace(
        workspace=workspace_path,
        pack_name=pack_name,
        seeds=seed_values,
        budget=effective_budget,
        primordia_root=Path(primordia_root),
        topograph_root=Path(topograph_root),
    )
    for key in (
        "workspace",
        "pack_path",
        "primordia_run_dir",
        "direct_seed_artifact",
        "staged_seed_artifact",
        "transfer_summary_report",
        "transfer_summary_data",
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


def _parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return [int(item) for item in values]


def _parse_optional_csv_ints(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return _parse_csv_ints(raw)
