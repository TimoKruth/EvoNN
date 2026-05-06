"""CLI for auditing benchmark ladder packs."""

from __future__ import annotations

from pathlib import Path

import typer

from evonn_compare.orchestration.benchmark_audit import audit_benchmark_pack


def benchmark_audit(
    pack: str = typer.Option(..., "--pack", help="Parity pack name or YAML path to audit"),
    output: str = typer.Option("benchmark_audit.md", "--output", help="Markdown or JSON audit output path"),
) -> None:
    """Audit benchmark support, contender floor metadata, and lane admission status."""

    result = audit_benchmark_pack(pack_name=pack, output=Path(output))
    output_path = Path(output)
    json_path = output_path if output_path.suffix == ".json" else output_path.with_suffix(".json")
    md_path = output_path.with_suffix(".md") if output_path.suffix == ".json" else output_path
    typer.echo(f"pack\t{result['pack_name']}")
    typer.echo(f"status\t{result['summary']['audit_status']}")
    typer.echo(f"benchmarks\t{result['benchmark_count']}")
    typer.echo(f"blocked\t{result['summary']['blocked_count']}")
    typer.echo(f"exploratory\t{result['summary']['exploratory_count']}")
    typer.echo(f"report\t{md_path.resolve()}")
    typer.echo(f"report_json\t{json_path.resolve()}")
