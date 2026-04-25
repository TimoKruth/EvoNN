"""Typer CLI for contender runs."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_contenders.benchmarks import list_benchmarks
from evonn_contenders.config import load_config
from evonn_contenders.export import export_symbiosis_contract, write_report
from evonn_contenders.pipeline import materialize_baseline_run, run_contenders
from evonn_contenders.storage import RunStore


app = typer.Typer(no_args_is_help=True, add_completion=False)
symbiosis_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(symbiosis_app, name="symbiosis")


@app.command()
def benchmarks(config: Path | None = typer.Option(default=None, exists=True, dir_okay=False, file_okay=True)) -> None:
    """List available benchmarks or those in config."""
    if config is None:
        for spec in list_benchmarks():
            typer.echo(f"{spec.name}\t{spec.task}\t{spec.metric_name}")
        return
    run_config = load_config(config)
    for name in run_config.benchmark_pool.benchmarks:
        typer.echo(name)


@app.command("run")
def run_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    run_dir: Path | None = typer.Option(default=None),
) -> None:
    """Run contender pool on config benchmarks."""
    run_config = load_config(config)
    resolved_run_dir = run_dir or Path("runs") / (run_config.run_name or f"{config.stem}_seed{run_config.seed}")
    path = run_contenders(run_config, run_dir=resolved_run_dir, config_path=config)
    typer.echo(str(path))


@app.command("materialize")
def materialize_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    run_dir: Path | None = typer.Option(default=None),
) -> None:
    """Materialize one run directory from baseline cache only."""

    run_config = load_config(config)
    resolved_run_dir = run_dir or Path("runs") / (run_config.run_name or f"{config.stem}_seed{run_config.seed}")
    try:
        path = materialize_baseline_run(run_config, run_dir=resolved_run_dir, config_path=config)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(path))


@app.command()
def report(run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True)) -> None:
    """Write markdown report."""
    typer.echo(str(write_report(run_dir)))


@app.command()
def inspect(run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True)) -> None:
    """Print compact run summary."""
    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    if not runs:
        store.close()
        raise typer.Exit(code=1)
    run = runs[0]
    results = store.load_results(run["run_id"])
    contenders = store.load_contenders(run["run_id"])
    store.close()
    typer.echo(f"run_id={run['run_id']}")
    typer.echo(f"benchmarks={len(results)}")
    typer.echo(f"contender_evals={len(contenders)}")
    typer.echo(f"statuses={','.join(sorted({record['status'] for record in results}))}")


@symbiosis_app.command("export")
def symbiosis_export(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    pack_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    output_dir: Path | None = typer.Option(default=None),
) -> None:
    """Export compare manifest/results."""
    manifest_path, results_path = export_symbiosis_contract(run_dir, pack_path, output_dir)
    typer.echo(f"{manifest_path}\n{results_path}")
