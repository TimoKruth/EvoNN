"""CLI compare command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.reporting.compare_md import render_comparison_markdown


def compare(
    left: str = typer.Argument(..., help="Left run directory"),
    right: str = typer.Argument(..., help="Right run directory"),
    pack: str = typer.Option(..., "--pack", help="Parity pack name or YAML path"),
    output: str | None = typer.Option(None, "--output", help="Optional markdown output path"),
) -> None:
    """Compare two exported runs."""

    pack_def = load_parity_pack(pack)
    left_ingestor = SystemIngestor(Path(left))
    right_ingestor = SystemIngestor(Path(right))
    result = ComparisonEngine().compare(
        left_manifest=left_ingestor.load_manifest(),
        left_results=left_ingestor.load_results(),
        right_manifest=right_ingestor.load_manifest(),
        right_results=right_ingestor.load_results(),
        pack=pack_def,
    )
    markdown = render_comparison_markdown(result)
    if output is not None:
        output_path = Path(output)
        json_output_path = output_path.with_suffix(".json")
        output_path.write_text(markdown, encoding="utf-8")
        json_output_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        typer.echo(f"report\t{output_path}")
        typer.echo(f"report_json\t{json_output_path}")
    typer.echo(markdown)
