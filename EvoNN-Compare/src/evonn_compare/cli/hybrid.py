"""Hybrid run commands."""

from __future__ import annotations

from pathlib import Path

import typer


def run(
    pack: str = typer.Option(..., "--pack", help="Parity pack name or YAML path"),
    seed: int = typer.Option(42, "--seed"),
    population: int = typer.Option(8, "--population"),
    generations: int = typer.Option(3, "--generations"),
    epochs: int = typer.Option(20, "--epochs"),
    output: str = typer.Option(..., "--output", help="Hybrid run output directory"),
) -> None:
    """Run hybrid evolution on a parity pack and export results."""

    from evonn_compare.contracts.parity import load_parity_pack
    from evonn_compare.hybrid.benchmarks import load_parity_pack_benchmarks
    from evonn_compare.hybrid.engine import HybridConfig, HybridEngine
    from evonn_compare.hybrid.export import export_hybrid_results

    pack_def = load_parity_pack(pack)
    output_dir = Path(output)
    engine = HybridEngine(
        HybridConfig(
            seed=seed,
            population_size=population,
            generations=generations,
            epochs=epochs,
        ),
        run_dir=output_dir,
    )
    benchmarks = load_parity_pack_benchmarks(pack, seed=seed)
    engine.run(benchmarks)
    export_hybrid_results(engine, output_dir, pack_def.name)
    typer.echo(f"hybrid\t{output_dir}")
