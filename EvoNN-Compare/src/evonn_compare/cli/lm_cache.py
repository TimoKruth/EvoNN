"""CLI for shared language-modeling cache generation and validation."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from evonn_shared.lm_cache import (
    LMCacheSpec,
    default_lm_cache_dir,
    default_lm_cache_spec,
    generate_lm_cache,
    validate_default_lm_cache,
)


def lm_cache(
    datasets: str = typer.Option(
        "tinystories_lm,wikitext2_lm",
        "--datasets",
        help="Comma-separated canonical LM cache datasets to generate or validate",
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", help="LM cache output directory"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Validate existing caches without generating"),
    context_length: int = typer.Option(256, "--context-length", help="Byte-token context length"),
    train_windows: int = typer.Option(4096, "--train-windows", help="Training windows to write"),
    val_windows: int = typer.Option(512, "--val-windows", help="Validation windows to write"),
    stride: int = typer.Option(128, "--stride", help="Window stride in bytes"),
    json_output: str | None = typer.Option(None, "--json-output", help="Optional JSON report path"),
) -> None:
    """Generate or validate shared byte-level real-text LM caches."""

    dataset_names = [part.strip() for part in datasets.split(",") if part.strip()]
    cache_dir = Path(output_dir) if output_dir is not None else default_lm_cache_dir()
    reports = []
    for dataset in dataset_names:
        if validate_only:
            report = validate_default_lm_cache(dataset, cache_dir=cache_dir)
        else:
            default_spec = default_lm_cache_spec(dataset)
            spec = LMCacheSpec(
                dataset=dataset,
                source=default_spec.source,
                source_kind=default_spec.source_kind,
                zip_member=default_spec.zip_member,
                context_length=context_length,
                train_windows=train_windows,
                val_windows=val_windows,
                stride=stride,
                vocab_size=default_spec.vocab_size,
                max_source_bytes=default_spec.max_source_bytes,
            )
            report = generate_lm_cache(spec, output_dir=cache_dir)
        reports.append(report)
        status = "ok" if report.get("ok") else "blocked"
        blockers = "; ".join(str(item) for item in report.get("blockers", [])) or "none"
        typer.echo(f"{dataset}\t{status}\t{report.get('path')}\t{blockers}")

    payload = {"cache_dir": str(cache_dir.resolve()), "reports": reports}
    if json_output is not None:
        output_path = Path(json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"report_json\t{output_path.resolve()}")
