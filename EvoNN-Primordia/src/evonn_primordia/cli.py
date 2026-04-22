from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evonn_primordia.config import load_config
from evonn_primordia.export import export_symbiosis_contract, write_report, write_seed_candidates
from evonn_primordia.export.report import (
    build_primitive_bank_summary,
    enrich_best_results,
    load_best_results,
    load_runtime_metadata,
)
from evonn_primordia.pipeline import run_search

app = typer.Typer(help="Primitive-first evolutionary search for EvoNN.", no_args_is_help=False)
symbiosis_app = typer.Typer(help="Export Primordia runs for compare tooling.", no_args_is_help=True)
seed_app = typer.Typer(help="Build benchmark-conditioned seed artifacts for later EvoNN systems.", no_args_is_help=True)
app.add_typer(symbiosis_app, name="symbiosis")
app.add_typer(seed_app, name="seed")
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Show a compact overview when called without a subcommand."""
    if ctx.invoked_subcommand is not None:
        return
    table = Table(title="Primordia")
    table.add_column("Area")
    table.add_column("Status")
    table.add_row("Package scaffold", "ready")
    table.add_row("MLX primitive lane", "ready")
    table.add_row("Fair export contract", "ready")
    table.add_row("Primitive bank export", "ready")
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

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_meta = load_runtime_metadata(summary)
    trial_records_path = run_dir / "trial_records.json"
    trial_records = json.loads(trial_records_path.read_text(encoding="utf-8")) if trial_records_path.exists() else []
    best_results = enrich_best_results(load_best_results(run_dir, summary), trial_records)
    primitive_bank_path = run_dir / "primitive_bank_summary.json"
    if primitive_bank_path.exists():
        primitive_bank = json.loads(primitive_bank_path.read_text(encoding="utf-8"))
    else:
        primitive_bank = build_primitive_bank_summary(
            summary=summary,
            best_results=best_results,
            trial_records=trial_records,
        )

    overview = Table(title=f"Run: {summary.get('run_name') or summary.get('run_id') or run_dir.name}")
    overview.add_column("Metric", style="cyan")
    overview.add_column("Value", style="green")
    overview.add_row("Runtime", runtime_meta["runtime"])
    overview.add_row("Runtime Version", runtime_meta["runtime_version"])
    overview.add_row("Precision Mode", runtime_meta["precision_mode"])
    overview.add_row("Evaluation Count", str(summary.get("evaluation_count", 0)))
    overview.add_row("Target Evaluations", str(summary.get("target_evaluation_count", "n/a")))
    overview.add_row("Benchmarks", str(summary.get("benchmark_count", 0)))
    overview.add_row("Failure Count", str(summary.get("failure_count", 0)))
    wall_clock = summary.get("wall_clock_seconds")
    if wall_clock is not None:
        overview.add_row("Wall Clock", f"{float(wall_clock):.1f}s")
    console.print(overview)

    usage = summary.get("primitive_usage") or {}
    if usage:
        usage_table = Table(title="Primitive Usage")
        usage_table.add_column("Family", style="cyan")
        usage_table.add_column("Evaluations", style="green")
        for family, count in usage.items():
            usage_table.add_row(str(family), str(count))
        console.print(usage_table)

    group_counts = summary.get("group_counts") or {}
    nonzero_groups = [(str(group), int(count)) for group, count in group_counts.items() if int(count) > 0]
    if nonzero_groups:
        group_table = Table(title="Benchmark Group Coverage")
        group_table.add_column("Group", style="cyan")
        group_table.add_column("Benchmarks", style="green")
        for group, count in nonzero_groups:
            group_table.add_row(group, str(count))
        console.print(group_table)

    bank_rows = primitive_bank.get("primitive_families") or []
    if bank_rows:
        bank_table = Table(title="Primitive Bank")
        bank_table.add_column("Family", style="cyan")
        bank_table.add_column("Evaluations", style="green")
        bank_table.add_column("Benchmark Wins", style="green")
        bank_table.add_column("Won Benchmarks", style="white")
        bank_table.add_column("Representative Architecture", style="white")
        for row in bank_rows[:8]:
            won = row.get("benchmarks_won") or row.get("won_benchmarks") or []
            bank_table.add_row(
                str(row.get("family", "unknown")),
                str(row.get("evaluation_count", 0)),
                str(row.get("benchmark_wins", 0)),
                ", ".join(map(str, won)) if won else "—",
                str(row.get("representative_architecture_summary") or "—"),
            )
        console.print(bank_table)

    if summary.get("failure_count", 0) and trial_records_path.exists():
        failures = [record for record in trial_records if record.get("status") != "ok"]
        if failures:
            failure_table = Table(title="Recent Failures")
            failure_table.add_column("Benchmark", style="cyan")
            failure_table.add_column("Primitive", style="green")
            failure_table.add_column("Reason", style="white")
            for record in failures[:5]:
                failure_table.add_row(
                    str(record.get("benchmark_name", "unknown")),
                    str(record.get("primitive_name", "unknown")),
                    str(record.get("failure_reason") or "unknown"),
                )
            console.print(failure_table)

    if best_results:
        best_table = Table(title="Best Benchmarks")
        best_table.add_column("Benchmark", style="cyan")
        best_table.add_column("Primitive", style="green")
        best_table.add_column("Metric", style="white")
        best_table.add_column("Value", style="green")
        best_table.add_column("Status", style="white")
        for row in best_results[:8]:
            value = row.get("metric_value")
            rendered_value = "---" if value is None else f"{float(value):.6f}"
            best_table.add_row(
                str(row.get("benchmark_name", "unknown")),
                str(row.get("primitive_name", "unknown")),
                str(row.get("metric_name", "metric")),
                rendered_value,
                str(row.get("status", "unknown")),
            )
        console.print(best_table)


@symbiosis_app.command("export")
def symbiosis_export(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    pack_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    output_dir: Path | None = typer.Option(default=None),
) -> None:
    """Export compare manifest/results."""

    manifest_path, results_path = export_symbiosis_contract(run_dir, pack_path, output_dir)
    typer.echo(f"{manifest_path}\n{results_path}")


@seed_app.command("export")
def seed_export(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    """Write Primordia seed candidates for later package seeding experiments."""

    typer.echo(str(write_seed_candidates(run_dir)))


if __name__ == "__main__":
    app()
