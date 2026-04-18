"""Typer CLI for Prism — family-based evolutionary NAS."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from prism.benchmarks.datasets import get_benchmark, list_benchmarks
from prism.benchmarks.parity import get_canonical_id, load_parity_pack, resolve_pack_path
from prism.config import load_config

console = Console()
app = typer.Typer(name="prism", help="Family-based evolutionary NAS")


# ===========================================================================
# Top-level commands
# ===========================================================================


@app.command("benchmarks")
def benchmarks_cmd() -> None:
    """List available benchmarks."""
    names = list_benchmarks()
    if not names:
        console.print("[yellow]No benchmarks found in catalog.[/yellow]")
        return

    table = Table(title="Available Benchmarks")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Input", style="white")
    table.add_column("Output", style="white")
    table.add_column("Source", style="dim")

    for name in names:
        try:
            spec = get_benchmark(name)
            input_str = str(spec.input_dim or spec.input_shape or "?")
            output_str = str(spec.num_classes or spec.output_dim or "?")
            table.add_row(name, spec.task, input_str, output_str, spec.source)
        except FileNotFoundError:
            table.add_row(name, "?", "?", "?", "?")

    console.print(table)


@app.command()
def evolve(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="Run output directory"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
) -> None:
    """Run evolution."""
    cfg = load_config(config)

    if run_dir is None:
        run_dir = str(Path("runs") / f"evolve-{cfg.seed}-{uuid.uuid4().hex[:6]}")

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # Save config to run directory
    config_dest = run_path / "config.yaml"
    if not config_dest.exists():
        config_dest.write_text(
            yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    console.print("[bold cyan]Prism Evolution[/bold cyan]")
    console.print(f"  Config: {config}")
    console.print(f"  Run dir: {run_path}")
    console.print(f"  Seed: {cfg.seed}")
    console.print(f"  Pack: {cfg.benchmark_pack.pack_name}")
    console.print(
        f"  Population: {cfg.evolution.population_size} | "
        f"Generations: {cfg.evolution.num_generations}"
    )

    if resume:
        console.print("[yellow]Resuming from checkpoint...[/yellow]")

    try:
        pack_path = resolve_pack_path(cfg.benchmark_pack.pack_name)
        benchmark_specs = load_parity_pack(pack_path)
        from prism.pipeline.coordinator import run_evolution
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except ImportError as e:
        console.print(f"[red]Pipeline import failed: {e}[/red]")
        raise typer.Exit(1)

    run_evolution(
        config=cfg,
        benchmark_specs=benchmark_specs,
        run_dir=str(run_path),
        resume=resume,
    )


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="Path to run directory"),
) -> None:
    """Generate markdown report for a completed run."""
    from prism.export.report import generate_report

    run_path = Path(run_dir)
    if not run_path.exists():
        run_path = Path("runs") / run_dir
    if not run_path.exists():
        console.print(f"[red]Run directory not found: {run_dir}[/red]")
        raise typer.Exit(1)

    text = generate_report(run_path)
    console.print(text)

    report_path = run_path / "report.md"
    report_path.write_text(text, encoding="utf-8")
    console.print(f"\n[green]Report saved to {report_path}[/green]")


@app.command()
def inspect(
    run_dir: str = typer.Argument(..., help="Path to run directory"),
) -> None:
    """Inspect run metrics."""
    from prism.genome import ModelGenome
    from prism.storage import RunStore

    run_path = Path(run_dir)
    if not run_path.exists():
        run_path = Path("runs") / run_dir
    if not run_path.exists():
        console.print(f"[red]Run directory not found: {run_dir}[/red]")
        raise typer.Exit(1)

    store = RunStore(run_path / "metrics.duckdb")
    run_id = _resolve_run_id(store)

    latest_gen = store.latest_generation(run_id)
    evaluations = store.load_evaluations(run_id)
    genome_rows = store.load_genomes(run_id)
    best_per_benchmark = store.load_best_per_benchmark(run_id)
    store.close()

    genomes: list[ModelGenome] = []
    for row in genome_rows:
        try:
            genomes.append(ModelGenome.model_validate(row))
        except Exception:
            pass

    table = Table(title=f"Run: {run_path.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Generation", str(latest_gen if latest_gen is not None else "N/A"))
    table.add_row("Genomes", str(len(genomes)))
    table.add_row("Total Evaluations", str(len(evaluations)))
    table.add_row("Benchmarks", str(len(best_per_benchmark)))

    # Best quality per benchmark
    if best_per_benchmark:
        qualities = [v["quality"] for v in best_per_benchmark.values() if v.get("quality")]
        if qualities:
            table.add_row("Best Quality (avg)", f"{sum(qualities) / len(qualities):.6f}")
            table.add_row("Best Quality (max)", f"{max(qualities):.6f}")

    # Family distribution
    if genomes:
        from collections import Counter
        families = Counter(g.family for g in genomes)
        table.add_row("Families", ", ".join(f"{f}({c})" for f, c in families.most_common()))

    console.print(table)

    # Per-benchmark table
    if best_per_benchmark:
        bench_table = Table(title="Per-Benchmark Best")
        bench_table.add_column("Benchmark", style="cyan")
        bench_table.add_column("Quality", style="green")
        bench_table.add_column("Metric", style="white")
        bench_table.add_column("Params", style="white")

        for bid, best in sorted(best_per_benchmark.items()):
            bench_table.add_row(
                bid,
                f"{best.get('quality', 0):.6f}",
                best.get("metric_name", "?"),
                str(best.get("parameter_count", "?")),
            )
        console.print(bench_table)


@app.command("analyze-compare")
def analyze_compare(
    summaries: list[str] = typer.Argument(..., help="Paths to four_way_summary.md files"),
    output: Optional[str] = typer.Option(None, "--output", help="Optional markdown output path"),
) -> None:
    """Aggregate compare summary markdown files into one Prism-oriented analysis."""
    from prism.analysis.compare import render_compare_analysis

    text = render_compare_analysis(summaries)
    console.print(text)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        console.print(f"\n[green]Analysis saved to {output_path}[/green]")


@app.command("analyze-matrix")
def analyze_matrix(
    matrix_root: str = typer.Argument(..., help="Path to compare matrix root directory"),
    output: Optional[str] = typer.Option(None, "--output", help="Optional markdown output path"),
) -> None:
    """Aggregate compare summaries and Prism run dirs from one matrix root."""
    from prism.analysis.matrix import render_matrix_analysis

    text = render_matrix_analysis(matrix_root)
    console.print(text)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        console.print(f"\n[green]Matrix analysis saved to {output_path}[/green]")


# ===========================================================================
# Symbiosis subcommand group
# ===========================================================================

symbiosis_app = typer.Typer(help="Symbiosis export commands")
app.add_typer(symbiosis_app, name="symbiosis")


@symbiosis_app.command("export")
def symbiosis_export(
    run_dir: str = typer.Argument(..., help="Run directory path"),
    pack: str = typer.Option(..., "--pack", help="Parity pack YAML path"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Output directory"),
) -> None:
    """Export symbiosis contract (manifest.json + results.json)."""
    from prism.export.symbiosis import export_symbiosis_contract

    run_path = Path(run_dir)
    if not run_path.exists():
        run_path = Path("runs") / run_dir

    manifest_path, results_path = export_symbiosis_contract(
        run_dir=run_path,
        pack_path=pack,
        output_dir=output_dir,
    )

    console.print(f"[green]manifest[/green]\t{manifest_path}")
    console.print(f"[green]results[/green]\t{results_path}")


# ===========================================================================
# Suite subcommand group
# ===========================================================================

suite_app = typer.Typer(help="Benchmark suite commands")
app.add_typer(suite_app, name="suite")


@suite_app.command("list")
def suite_list(
    task: Optional[str] = typer.Option(None, "--task", help="Filter by task (classification/regression)"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source (sklearn/openml/image)"),
) -> None:
    """List datasets in the benchmark catalog."""
    names = list_benchmarks()
    if not names:
        console.print("[yellow]No benchmarks found in catalog.[/yellow]")
        return

    table = Table(title="Benchmark Catalog")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Source", style="white")
    table.add_column("Input Dim", style="white")
    table.add_column("Classes", style="white")
    table.add_column("Canonical ID", style="dim")

    for name in names:
        try:
            spec = get_benchmark(name)
            if task and spec.task != task:
                continue
            if source and spec.source != source:
                continue
            table.add_row(
                name,
                spec.task,
                spec.source,
                str(spec.input_dim or "?"),
                str(spec.num_classes or "?"),
                get_canonical_id(name),
            )
        except FileNotFoundError:
            pass

    console.print(table)


@suite_app.command("info")
def suite_info(
    name: str = typer.Argument(..., help="Benchmark name"),
) -> None:
    """Show detailed info for a benchmark."""
    try:
        spec = get_benchmark(name)
    except FileNotFoundError:
        console.print(f"[red]Benchmark '{name}' not found.[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Benchmark: {name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("ID", spec.id)
    table.add_row("Task", spec.task)
    table.add_row("Source", spec.source)
    table.add_row("Dataset", spec.dataset or "N/A")
    table.add_row("Input Dim", str(spec.input_dim or spec.input_shape or "?"))
    table.add_row("Output Dim", str(spec.output_dim))
    table.add_row("Metric", f"{spec.metric_name} ({spec.metric_direction})")
    table.add_row("Canonical ID", get_canonical_id(name))

    if spec.source_id:
        table.add_row("OpenML ID", str(spec.source_id))
    if spec.n_samples != 1000:
        table.add_row("Samples", str(spec.n_samples))

    console.print(table)


# ===========================================================================
# Helpers
# ===========================================================================


def _resolve_run_id(store) -> str:
    """Resolve run_id from the store."""
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "default"
