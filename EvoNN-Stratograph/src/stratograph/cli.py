"""Typer CLI for Stratograph."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stratograph.analysis import analyze_run_motifs, run_ablation_matrix, run_ablation_suite
from stratograph.benchmarks import list_benchmarks
from stratograph.benchmarks.lm import available_lm_caches, warm_lm_cache
from stratograph.config import load_config
from stratograph.export import export_symbiosis_contract, write_report
from stratograph.export.report import (
    load_report_context,
    load_runtime_metadata,
    summarize_failure_patterns,
)
from stratograph.pipeline import build_execution_ladder, run_evolution, run_execution_ladder


app = typer.Typer(no_args_is_help=True, add_completion=False)
symbiosis_app = typer.Typer(no_args_is_help=True, add_completion=False)
motifs_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(symbiosis_app, name="symbiosis")
app.add_typer(motifs_app, name="motifs")
console = Console()


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


@app.command("warm-cache")
def warm_cache(
    config: Path | None = typer.Option(default=None, exists=True, dir_okay=False, file_okay=True),
    dataset: list[str] | None = typer.Option(default=None, help="Explicit LM dataset ids."),
    overwrite: bool = typer.Option(default=False, help="Overwrite existing local repo cache files."),
) -> None:
    """Warm LM caches into Stratograph repo cache."""
    datasets = list(dataset or [])
    if config is not None:
        run_config = load_config(config)
        datasets.extend(name for name in run_config.benchmark_pool.benchmarks if "_lm" in name or name == "tiny_lm_synthetic")
    datasets = [name for name in datasets if name != "tiny_lm_synthetic"]
    copied = warm_lm_cache(sorted(set(datasets)) or None, overwrite=overwrite)
    typer.echo("\n".join(str(path) for path in copied))


@app.command("list-lm-caches")
def list_lm_caches() -> None:
    """List resolvable LM caches."""
    for name in available_lm_caches():
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
    try:
        context = load_report_context(run_dir)
    except ValueError as exc:
        raise typer.Exit(code=1) from exc
    run = context["run"]
    results = context["results"]
    genomes = context["genomes"]
    budget_meta = context["budget_meta"]
    status_payload = context["status"]
    runtime_meta = load_runtime_metadata(budget_meta)
    non_ok_results = context["non_ok_results"]
    failed_results = context["failed_results"]
    skipped_results = context["skipped_results"]
    best_results = context["best_results"]
    representative_genome = context["representative_genome"]

    overview = Table(title="Run Overview")
    overview.add_column("Metric", style="cyan")
    overview.add_column("Value", style="green")
    overview.add_row("Run ID", str(run["run_id"]))
    overview.add_row("Seed", str(run["seed"]))
    overview.add_row("Created At", str(budget_meta.get("created_at") or run.get("created_at") or "unknown"))
    overview.add_row("Run State", str(status_payload.get("state", "unknown")))
    overview.add_row("Benchmarks", str(len(results)))
    overview.add_row("Genomes Stored", str(len(genomes)))
    overview.add_row("Runtime", runtime_meta["runtime_backend"])
    overview.add_row("Runtime Version", runtime_meta["runtime_version"])
    overview.add_row("Precision Mode", runtime_meta["precision_mode"])
    overview.add_row("Architecture Mode", str(budget_meta.get("architecture_mode", "unknown")))
    overview.add_row("Evaluation Count", str(budget_meta.get("evaluation_count", 0)))
    overview.add_row("Effective Training Epochs", str(budget_meta.get("effective_training_epochs", "unknown")))
    overview.add_row("Wall Clock Seconds", f"{float(budget_meta.get('wall_clock_seconds', 0.0)):.3f}")
    overview.add_row(
        "Completed Benchmarks",
        f"{status_payload.get('completed_count', len(context['ok_results']) + len(skipped_results) + len(failed_results))}/{status_payload.get('total_benchmarks', len(results))}",
    )
    overview.add_row("Remaining Benchmarks", str(status_payload.get("remaining_count", 0)))
    overview.add_row("Novelty Mean", f"{float(budget_meta.get('novelty_score_mean', 0.0)):.4f}")
    overview.add_row("Occupied Niches", str(budget_meta.get("map_elites_occupied_niches", 0)))
    if representative_genome is not None:
        overview.add_row("Representative Genome", str(representative_genome.genome_id))
        overview.add_row("Cell Library Size", str(len(representative_genome.cell_library)))
        overview.add_row("Macro Depth", str(representative_genome.macro_depth))
        overview.add_row("Avg Cell Depth", f"{representative_genome.average_cell_depth:.2f}")
        overview.add_row("Reuse Ratio", f"{representative_genome.reuse_ratio:.4f}")
    overview.add_row(
        "Status Mix",
        f"ok={len(context['ok_results'])}, skipped={len(skipped_results)}, failed={len(failed_results)}",
    )
    checkpoint_path = run_dir / "checkpoint.json"
    if context["status_path"].exists():
        overview.add_row("Status Artifact", str(context["status_path"]))
    if checkpoint_path.exists():
        overview.add_row("Checkpoint", str(checkpoint_path))
    console.print(overview)

    best_table = Table(title="Best Benchmarks")
    best_table.add_column("Benchmark", style="cyan")
    best_table.add_column("Metric", style="white")
    best_table.add_column("Value", style="green")
    best_table.add_column("Quality", style="green")
    best_table.add_column("Genome", style="white")
    best_table.add_column("Architecture", style="white")
    if best_results:
        for record in best_results:
            metric_value = record.get("metric_value")
            quality = record.get("quality")
            best_table.add_row(
                str(record["benchmark_name"]),
                str(record["metric_name"]),
                "---" if metric_value is None else f"{float(metric_value):.6f}",
                "---" if quality is None else f"{float(quality):.4f}",
                str(record.get("genome_id") or "—"),
                str(record.get("architecture_summary") or "—"),
            )
    else:
        best_table.add_row("none", "—", "---", "---", "—", "—")
    console.print(best_table)

    failure_patterns = summarize_failure_patterns(non_ok_results)

    failure_table = Table(title="Failure Patterns")
    failure_table.add_column("Reason", style="white")
    failure_table.add_column("Count", style="green")
    if failure_patterns:
        for reason, count in failure_patterns:
            failure_table.add_row(reason, str(count))
    else:
        failure_table.add_row("none", "0")
    console.print(failure_table)

    failure_detail_table = Table(title="Failure Details")
    failure_detail_table.add_column("Benchmark", style="cyan")
    failure_detail_table.add_column("Reason", style="white")
    if failed_results:
        for record in failed_results:
            failure_detail_table.add_row(
                str(record["benchmark_name"]),
                str(record.get("failure_reason") or "unknown"),
            )
    else:
        failure_detail_table.add_row("none", "no failed benchmarks")
    console.print(failure_detail_table)


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


@app.command("ablate-matrix")
def ablate_matrix(
    config: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    workspace: Path = typer.Option(..., help="Workspace for matrix outputs."),
) -> None:
    """Run pack matrix for hierarchy/value study."""
    report_path = run_ablation_matrix(load_config(config), workspace=workspace, config_path=config)
    typer.echo(str(report_path))


@motifs_app.command("analyze")
def motifs_analyze(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Analyze repeated motifs from winning genomes in run dir."""
    typer.echo(str(analyze_run_motifs(run_dir)))
