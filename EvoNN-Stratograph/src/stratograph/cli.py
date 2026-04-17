"""Typer CLI for Stratograph."""

from __future__ import annotations

from pathlib import Path

import typer

from stratograph.analysis import analyze_run_motifs, run_ablation_suite
from stratograph.benchmarks import list_benchmarks
from stratograph.config import load_config
from stratograph.export import export_symbiosis_contract, write_report
from stratograph.pipeline import build_execution_ladder, run_evolution, run_execution_ladder
from stratograph.storage import RunStore


app = typer.Typer(no_args_is_help=True, add_completion=False)
symbiosis_app = typer.Typer(no_args_is_help=True, add_completion=False)
motifs_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(symbiosis_app, name="symbiosis")
app.add_typer(motifs_app, name="motifs")


@app.command()
def benchmarks(config: Path | None = typer.Option(default=None, exists=True, dir_okay=False, file_okay=True)) -> None:
    """List available benchmarks or those from config."""
    if config is None:
        for spec in list_benchmarks():
            typer.echo(f"{spec.name}\t{spec.task}\t{spec.metric_name}")
        return

    run_config = load_config(config)
    for name in run_config.benchmark_pool.benchmarks:
        typer.echo(name)


@app.command()
def evolve(
    config: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    run_dir: Path | None = typer.Option(default=None),
    resume: bool = typer.Option(default=False, help="Resume partial run from checkpoint if present."),
) -> None:
    """Run Stratograph evolution."""
    run_config = load_config(config)
    resolved_run_dir = run_dir or Path("runs") / (run_config.run_name or f"{config.stem}_seed{run_config.seed}")
    path = run_evolution(run_config, run_dir=resolved_run_dir, config_path=config, resume=resume)
    typer.echo(str(path))


@app.command()
def report(run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True)) -> None:
    """Write report markdown for run dir."""
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
    genomes = store.load_genomes(run["run_id"])
    store.close()
    typer.echo(f"run_id={run['run_id']}")
    typer.echo(f"benchmarks={len(results)}")
    typer.echo(f"genomes={len(genomes)}")
    typer.echo(f"statuses={','.join(sorted({record['status'] for record in results}))}")
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_path.exists():
        typer.echo(f"checkpoint={checkpoint_path}")


@symbiosis_app.command("export")
def symbiosis_export(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    pack_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    output_dir: Path | None = typer.Option(default=None),
) -> None:
    """Export compare manifest/results."""
    manifest_path, results_path = export_symbiosis_contract(run_dir, pack_path, output_dir)
    typer.echo(f"{manifest_path}\n{results_path}")


@app.command()
def ladder(
    workspace: Path = typer.Option(default=Path("manual_compare_runs") / "execution_ladder"),
    execute: bool = typer.Option(default=True, help="Run ladder after generating configs."),
) -> None:
    """Generate and optionally run execution ladder cases."""
    if execute:
        manifests = run_execution_ladder(workspace)
        for manifest in manifests:
            typer.echo(str(manifest))
        return
    for case in build_execution_ladder(workspace):
        typer.echo(f"{case.name}\t{case.config_path}\t{case.pack_path}\t{case.run_dir}")


@app.command()
def ablate(
    config: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    workspace: Path = typer.Option(..., help="Workspace for ablation suite outputs."),
) -> None:
    """Run flat/unshared/shared hierarchy ablations."""
    report_path = run_ablation_suite(load_config(config), workspace=workspace, config_path=config)
    typer.echo(str(report_path))


@motifs_app.command("analyze")
def motifs_analyze(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Analyze repeated motifs from winning genomes in run dir."""
    typer.echo(str(analyze_run_motifs(run_dir)))
