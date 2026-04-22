"""Typer CLI for Topograph — topology-first evolutionary NAS."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from topograph.benchmarks.parity import get_benchmark, list_benchmarks
from topograph.config import RunConfig, load_config

console = Console()
app = typer.Typer(name="topograph", help="Topology-first evolutionary NAS")


# ===========================================================================
# Top-level commands
# ===========================================================================


@app.command()
def benchmarks() -> None:
    """List available benchmarks."""
    names = list_benchmarks()
    if not names:
        console.print("[yellow]No benchmarks found in catalog.[/yellow]")
        return

    table = Table(title="Available Benchmarks")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Input Dim", style="white")
    table.add_column("Classes", style="white")

    for name in names:
        try:
            spec = get_benchmark(name)
            table.add_row(
                name,
                spec.task,
                str(spec.input_dim or "?"),
                str(spec.num_classes or "?"),
            )
        except FileNotFoundError:
            table.add_row(name, "?", "?", "?")

    console.print(table)


@app.command()
def evolve(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="Run output directory"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
    parallel_workers: int | None = typer.Option(
        None, "--parallel-workers", help="Override training.parallel_workers",
    ),
    parallel_cpu_fraction_limit: float | None = typer.Option(
        None,
        "--parallel-cpu-fraction-limit",
        help="Override training.parallel_cpu_fraction_limit",
    ),
    parallel_memory_fraction_limit: float | None = typer.Option(
        None,
        "--parallel-memory-fraction-limit",
        help="Override training.parallel_memory_fraction_limit",
    ),
    parallel_reserved_system_memory_bytes: int | None = typer.Option(
        None,
        "--parallel-reserved-system-memory-bytes",
        help="Override training.parallel_reserved_system_memory_bytes",
    ),
    parallel_worker_thread_limit: int | None = typer.Option(
        None,
        "--parallel-worker-thread-limit",
        help="Override training.parallel_worker_thread_limit",
    ),
) -> None:
    """Run evolution."""
    cfg = load_config(config)
    cfg = _apply_evolve_overrides(
        cfg,
        parallel_workers=parallel_workers,
        parallel_cpu_fraction_limit=parallel_cpu_fraction_limit,
        parallel_memory_fraction_limit=parallel_memory_fraction_limit,
        parallel_reserved_system_memory_bytes=parallel_reserved_system_memory_bytes,
        parallel_worker_thread_limit=parallel_worker_thread_limit,
    )

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

    console.print(f"[bold cyan]Topograph Evolution[/bold cyan]")
    console.print(f"  Config: {config}")
    console.print(f"  Run dir: {run_path}")
    console.print(f"  Seed: {cfg.seed}")
    console.print(f"  Benchmark: {cfg.benchmark}")
    console.print(
        f"  Population: {cfg.evolution.population_size} | "
        f"Generations: {cfg.evolution.num_generations}"
    )

    if resume:
        console.print("[yellow]Resuming from checkpoint...[/yellow]")

    try:
        from topograph.pipeline.coordinator import run_evolution

        run_evolution(
            config=cfg,
            benchmark_spec=None if cfg.benchmark_pool else get_benchmark(cfg.benchmark),
            run_dir=str(run_path),
            resume=resume,
        )
    except ImportError as e:
        console.print(f"[red]Missing dependency for evolution: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="Path to run directory"),
) -> None:
    """Generate markdown report for a run."""
    from topograph.export.report import generate_report

    text = generate_report(run_dir)
    console.print(text)

    report_path = Path(run_dir) / "report.md"
    report_path.write_text(text, encoding="utf-8")
    console.print(f"\n[green]Report saved to {report_path}[/green]")


@app.command()
def inspect(
    run_dir: str = typer.Argument(..., help="Path to run directory"),
) -> None:
    """Inspect run metrics."""
    from topograph.export.report import dag_summary, load_report_context, primordia_seeding_rows

    run_path = Path(run_dir)
    context = load_report_context(run_path)
    latest_gen = context["latest_generation"]
    if latest_gen is None:
        console.print("[yellow]No generations found.[/yellow]")
        return

    run = context["run"]
    population = context["population"]
    best = context["best_genome"]
    budget = context["budget_meta"]
    best_results = context["best_results"]
    failed_results = context["failed_results"]
    skipped_results = context["skipped_results"]
    ok_results = context["ok_results"]
    checkpoint_path = context["checkpoint_path"]

    table = Table(title="Run Overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run ID", str(run.get("run_id") or context["run_id"] or run_path.name))
    table.add_row("Seed", str(run.get("seed", "unknown")))
    table.add_row("Generation", str(latest_gen))
    table.add_row("Population", str(len(population)))
    table.add_row(
        "Best Fitness",
        f"{best.fitness:.6f}" if best.fitness is not None else "N/A",
    )
    table.add_row("Best Layers", str(len(best.enabled_layers)))
    table.add_row("Best Connections", str(len(best.enabled_connections)))
    table.add_row("Best Params", str(best.param_count))
    table.add_row("Best Model Bytes", str(best.model_bytes))
    table.add_row("Evaluated Benchmarks", str(len(best_results)))
    table.add_row(
        "Status Mix",
        f"ok={len(ok_results)}, skipped={len(skipped_results)}, failed={len(failed_results)}",
    )

    summary = dag_summary(best)
    table.add_row("Best DAG Depth", str(summary["depth"]))
    table.add_row("Best Skip Connections", str(summary["skip_connections"]))
    table.add_row("Best Connectivity", f"{summary['connectivity_ratio']:.4f}")

    if budget.get("wall_clock_seconds"):
        table.add_row("Wall Clock", f"{budget['wall_clock_seconds']:.1f}s")
    if budget.get("evaluation_count"):
        table.add_row("Total Evaluations", str(budget["evaluation_count"]))
    if budget.get("effective_training_epochs") is not None:
        table.add_row("Effective Training Epochs", str(budget["effective_training_epochs"]))
    if budget.get("runtime_backend"):
        table.add_row("Runtime", str(budget["runtime_backend"]))
    if budget.get("runtime_version") is not None:
        table.add_row("Runtime Version", str(budget.get("runtime_version") or "unknown"))
    if budget.get("precision_mode") is not None:
        table.add_row("Precision Mode", str(budget.get("precision_mode") or "unknown"))
    if budget.get("novelty_score_mean") is not None:
        table.add_row("Novelty Mean", f"{float(budget['novelty_score_mean']):.4f}")
    if budget.get("map_elites_occupied_niches") is not None:
        table.add_row("Occupied Niches", str(budget.get("map_elites_occupied_niches", 0)))
    for label, value in primordia_seeding_rows(budget.get("primordia_seeding")):
        table.add_row(label, value)
    if checkpoint_path.exists():
        table.add_row("Checkpoint", str(checkpoint_path))

    console.print(table)

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

    failure_table = Table(title="Failure Details")
    failure_table.add_column("Benchmark", style="cyan")
    failure_table.add_column("Reason", style="white")
    if failed_results:
        for record in failed_results:
            failure_table.add_row(
                str(record["benchmark_name"]),
                str(record.get("failure_reason") or "unknown"),
            )
    else:
        failure_table.add_row("none", "no failed benchmarks")
    console.print(failure_table)


@app.command("export")
def export_cmd(
    run_dir: str = typer.Argument(..., help="Path to run directory"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Export run results as JSON."""
    from topograph.genome.codec import dict_to_genome
    from topograph.storage import RunStore

    run_path = Path(run_dir)
    store = RunStore(run_path / "metrics.duckdb")
    run_id = _resolve_run_id(store)

    latest_gen = store.load_latest_generation(run_id)
    if latest_gen is None:
        console.print("[yellow]No generations found.[/yellow]")
        store.close()
        return

    genome_dicts = store.load_genomes(run_id, latest_gen)
    population = [dict_to_genome(d) for d in genome_dicts]
    best = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
    budget = store.load_budget_metadata(run_id) or {}
    store.close()

    dest = Path(output) if output else run_path
    dest.mkdir(parents=True, exist_ok=True)

    summary = {
        "system": "topograph",
        "run_dir": str(run_path),
        "generation": latest_gen,
        "population_size": len(population),
        "best_fitness": best.fitness,
        "best_topology": {
            "layers": len(best.enabled_layers),
            "connections": len(best.enabled_connections),
        },
        "budget": budget,
    }

    out_path = dest / "export.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]Exported to {out_path}[/green]")


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
    """Export symbiosis contract."""
    from topograph.export.symbiosis import export_symbiosis_contract

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
    domain: Optional[str] = typer.Option(None, "--domain", help="Filter by domain"),
    task: Optional[str] = typer.Option(None, "--task", help="Filter by task"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
) -> None:
    """List datasets in registry."""
    from topograph.benchmarks.registry import DatasetRegistry

    registry = DatasetRegistry()
    datasets = registry.list(domain=domain, task=task, tag=tag)

    if not datasets:
        console.print("[yellow]No datasets found matching filters.[/yellow]")
        return

    table = Table(title="Dataset Registry")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Source", style="white")
    table.add_column("Domain", style="white")
    table.add_column("Tags", style="dim")

    for meta in datasets:
        table.add_row(
            meta.name,
            meta.task,
            meta.source.value,
            meta.domain,
            ", ".join(meta.tags) if meta.tags else "",
        )

    console.print(table)


@suite_app.command("baselines")
def suite_baselines(
    dataset: str = typer.Argument(..., help="Dataset name from catalog"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run baseline models on dataset."""
    try:
        from topograph.baselines.runner import format_results, run_baselines_for_dataset
    except ImportError:
        console.print(
            "[red]Baselines module not available. "
            "Install with: uv pip install topograph[baselines][/red]"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Running baselines on {dataset} (seed={seed})...[/cyan]")
    results = run_baselines_for_dataset(dataset, seed=seed)
    console.print(format_results(results))


# ===========================================================================
# Helpers
# ===========================================================================


def _resolve_run_id(store) -> str:
    """Resolve run_id from the store."""
    try:
        store.load_run("current")
        return "current"
    except ValueError:
        pass
    row = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return row[0]
    return "current"


def _apply_evolve_overrides(
    cfg: RunConfig,
    *,
    parallel_workers: int | None,
    parallel_cpu_fraction_limit: float | None,
    parallel_memory_fraction_limit: float | None,
    parallel_reserved_system_memory_bytes: int | None,
    parallel_worker_thread_limit: int | None,
) -> RunConfig:
    training_updates: dict[str, int | float] = {}
    if parallel_workers is not None:
        training_updates["parallel_workers"] = parallel_workers
    if parallel_cpu_fraction_limit is not None:
        training_updates["parallel_cpu_fraction_limit"] = parallel_cpu_fraction_limit
    if parallel_memory_fraction_limit is not None:
        training_updates["parallel_memory_fraction_limit"] = parallel_memory_fraction_limit
    if parallel_reserved_system_memory_bytes is not None:
        training_updates["parallel_reserved_system_memory_bytes"] = (
            parallel_reserved_system_memory_bytes
        )
    if parallel_worker_thread_limit is not None:
        training_updates["parallel_worker_thread_limit"] = parallel_worker_thread_limit
    if not training_updates:
        return cfg

    payload = cfg.model_dump(mode="json")
    payload.setdefault("training", {}).update(training_updates)
    return RunConfig.model_validate(payload)
