from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evonn_primordia.config import load_config
from evonn_primordia.export import export_symbiosis_contract, write_report
from evonn_primordia.pipeline import run_search

app = typer.Typer(help="Primitive-first evolutionary search for EvoNN.", no_args_is_help=False)
symbiosis_app = typer.Typer(help="Export Primordia runs for EvoNN-Compare.", no_args_is_help=True)
app.add_typer(symbiosis_app, name="symbiosis")
console = Console()


@app.callback(invoke_without_command=True)
def main() -> None:
    """Show a compact overview when called without a subcommand."""
    table = Table(title="Primordia")
    table.add_column("Area")
    table.add_column("Status")
    table.add_row("Package scaffold", "ready")
    table.add_row("Budget-matched primitive lane", "ready")
    table.add_row("Fair export contract", "ready")
    table.add_row("Motif bank export", "next")
    console.print(table)
    console.print("Run `primordia run --config ...` or inspect the roadmap in VISION.md / IMPLEMENTATION_PLAN.md.")


@app.command("run")
def run_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    run_dir: Path | None = typer.Option(default=None),
) -> None:
    """Run Primordia on a benchmark pack/config."""

    run_config = load_config(config)
    resolved_run_dir = run_dir or Path("runs") / (run_config.run_name or f"{config.stem}_seed{run_config.seed}")
    path = run_search(run_config, run_dir=resolved_run_dir, config_path=config)
    typer.echo(str(path))


@app.command()
def report(run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True)) -> None:
    """Write or refresh the markdown report."""

    typer.echo(str(write_report(run_dir)))


@app.command()
def inspect(run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True)) -> None:
    """Print a compact run summary."""

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise typer.Exit(code=1)
    typer.echo(summary_path.read_text(encoding="utf-8"))


@symbiosis_app.command("export")
def symbiosis_export(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    pack_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    output_dir: Path | None = typer.Option(default=None),
) -> None:
    """Export compare manifest/results."""

    manifest_path, results_path = export_symbiosis_contract(run_dir, pack_path, output_dir)
    typer.echo(f"{manifest_path}\n{results_path}")


if __name__ == "__main__":
    app()
