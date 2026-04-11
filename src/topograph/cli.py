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
from topograph.config import load_config

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

    # Import and run the evolution pipeline
    try:
        import copy
        import random
        import time

        from topograph.cache import WeightCache
        from topograph.genome.codec import dict_to_genome, genome_to_dict
        from topograph.genome.genome import Genome, InnovationCounter
        from topograph.monitor import TerminalMonitor
        from topograph.operators.crossover import crossover
        from topograph.operators.mutate import (
            mutate_activation,
            mutate_add_connection,
            mutate_add_layer,
            mutate_remove_connection,
            mutate_remove_layer,
            mutate_width,
        )
        from topograph.pipeline.evaluate import (
            GenerationState,
            evaluate,
            evaluate_pool,
            score,
        )
        from topograph.pipeline.select import rank_based_select
        from topograph.storage import RunStore

        spec = get_benchmark(cfg.benchmark)
        store = RunStore(run_path / "metrics.duckdb")
        monitor = TerminalMonitor()
        cache = WeightCache()
        counter = InnovationCounter()
        rng = random.Random(cfg.seed)

        run_id = store.save_run(None, cfg.model_dump(mode="json"))

        # Resume or initialize
        start_gen = 0
        if resume:
            latest = store.load_latest_generation(run_id)
            if latest is not None:
                start_gen = latest + 1
                saved_counter = store.load_innovation_counter(run_id)
                if saved_counter:
                    counter = InnovationCounter(saved_counter)
                genome_dicts = store.load_genomes(run_id, latest)
                population = [dict_to_genome(d) for d in genome_dicts]
                console.print(f"  Resumed from generation {latest}")
            else:
                population = [
                    Genome.create_seed(
                        counter, rng, mixed_precision=bool(cfg.quantization_schedule),
                    )
                    for _ in range(cfg.evolution.population_size)
                ]
        else:
            population = [
                Genome.create_seed(
                    counter, rng, mixed_precision=bool(cfg.quantization_schedule),
                )
                for _ in range(cfg.evolution.population_size)
            ]

        # Load benchmark pool specs if configured
        pool_specs = None
        if cfg.benchmark_pool:
            pool_specs = [get_benchmark(name) for name in cfg.benchmark_pool.benchmarks]

        # Main evolution loop
        total_evaluations = 0
        evo = cfg.evolution
        rates = evo.mutation_rates

        for gen in range(start_gen, evo.num_generations):
            state = GenerationState(
                generation=gen,
                population=population,
                total_evaluations=total_evaluations,
            )

            # Evaluate
            if pool_specs and cfg.benchmark_pool:
                state = evaluate_pool(state, cfg, pool_specs, cache=cache, rng=rng)
            else:
                state = evaluate(state, cfg, spec, cache=cache)

            # Score (complexity penalty, device targets)
            state = score(state, cfg)
            total_evaluations = state.total_evaluations

            # Monitor
            fitnesses = [f for f in state.fitnesses if f != float("inf")]
            if fitnesses:
                monitor.on_generation(
                    generation=gen,
                    total=evo.num_generations,
                    best_fitness=min(fitnesses),
                    avg_fitness=sum(fitnesses) / len(fitnesses),
                    worst_fitness=max(fitnesses),
                    population_size=len(population),
                )

            # Save to DB
            genome_dicts_save = [genome_to_dict(g) for g in state.population]
            store.save_genomes(run_id, gen, genome_dicts_save)
            store.save_innovation_counter(run_id, counter._n)

            # Select + reproduce for next generation
            if gen < evo.num_generations - 1:
                # Elites
                sorted_pop = sorted(
                    zip(state.fitnesses, state.population),
                    key=lambda x: x[0],
                )
                elites = [copy.deepcopy(g) for _, g in sorted_pop[: evo.elite_count]]

                # Fill rest via selection + crossover + mutation
                offspring: list[Genome] = []
                target = evo.population_size - len(elites)

                parents = rank_based_select(
                    state.population, state.fitnesses, target * 2, rng,
                )
                for i in range(0, len(parents) - 1, 2):
                    if len(offspring) >= target:
                        break
                    if rng.random() < evo.crossover_ratio:
                        child = crossover(parents[i], parents[i + 1], rng)
                    else:
                        child = copy.deepcopy(parents[i])

                    # Apply mutations
                    if rng.random() < rates.width:
                        child = mutate_width(child, rng)
                    if rng.random() < rates.activation:
                        child = mutate_activation(child, rng)
                    if rng.random() < rates.add_layer:
                        child = mutate_add_layer(child, counter, rng)
                    if rng.random() < rates.remove_layer:
                        child = mutate_remove_layer(child, rng)
                    if rng.random() < rates.add_connection:
                        child = mutate_add_connection(child, counter, rng)
                    if rng.random() < rates.remove_connection:
                        child = mutate_remove_connection(child, rng)

                    offspring.append(child)

                population = elites + offspring[: target]

        # Final summary
        best = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
        elapsed = time.time() - monitor.start_time

        store.save_budget_metadata(run_id, {
            "evaluation_count": total_evaluations,
            "wall_clock_seconds": round(elapsed, 2),
            "total_generations": evo.num_generations,
            "population_size": evo.population_size,
        })

        monitor.on_complete(
            best_fitness=best.fitness if best.fitness is not None else float("inf"),
            total_generations=evo.num_generations,
            elapsed=elapsed,
        )
        store.close()

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

    table = Table(title=f"Run: {run_path.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

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

    if budget.get("wall_clock_seconds"):
        table.add_row("Wall Clock", f"{budget['wall_clock_seconds']:.1f}s")
    if budget.get("evaluation_count"):
        table.add_row("Total Evaluations", str(budget["evaluation_count"]))

    console.print(table)
    store.close()


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
