"""Symbiosis-style export contract for Stratograph prototype runs."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import stratograph
from stratograph.benchmarks import get_benchmark
from stratograph.benchmarks.parity import fallback_native_id, load_parity_pack
from stratograph.config import load_config
from stratograph.export.report import write_report
from stratograph.storage import RunStore


def export_symbiosis_contract(
    run_dir: str | Path,
    pack_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export manifest/results contract for compare tooling."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(run_dir / "config.yaml")
    pack = load_parity_pack(pack_path)
    store = RunStore(run_dir / "metrics.duckdb")
    runs = store.load_runs()
    if not runs:
        store.close()
        raise ValueError(f"No stored runs in {run_dir}")
    run = runs[0]
    results = {record["benchmark_name"]: record for record in store.load_results(run["run_id"])}
    genomes = store.load_genomes(run["run_id"])
    budget_meta = store.load_budget_metadata(run["run_id"])
    store.close()

    report_path = write_report(run_dir)
    _write_summary_json(output_dir / "genome_summary.json", genomes)
    _write_summary_json(output_dir / "model_summary.json", results)
    dataset_manifest = []

    manifest_benchmarks: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    for entry in pack.benchmarks:
        native_name = fallback_native_id(entry)
        spec = get_benchmark(native_name)
        result = results.get(native_name)
        status = result["status"] if result else "missing"
        manifest_benchmarks.append(
            {
                "benchmark_id": entry.benchmark_id,
                "task_kind": entry.task_kind,
                "metric_name": entry.metric_name,
                "metric_direction": entry.metric_direction,
                "status": status,
            }
        )
        result_records.append(
            {
                "system": "stratograph",
                "run_id": run["run_id"],
                "benchmark_id": entry.benchmark_id,
                "metric_name": entry.metric_name,
                "metric_direction": entry.metric_direction,
                "metric_value": result["metric_value"] if result else None,
                "quality": result["quality"] if result else None,
                "parameter_count": result["parameter_count"] if result else None,
                "train_seconds": result["train_seconds"] if result else None,
                "peak_memory_mb": None,
                "architecture_summary": result["architecture_summary"] if result else None,
                "genome_id": result["genome_id"] if result else None,
                "status": status,
                "failure_reason": result["failure_reason"] if result else ("missing_result" if not result else None),
            }
        )
        dataset_manifest.append(
            {
                "benchmark_id": entry.benchmark_id,
                "native_name": native_name,
                "source": spec.source,
                "task": spec.task,
                "input_dim": spec.input_dim,
                "num_classes": spec.num_classes,
            }
        )
        del spec

    _write_summary_json(output_dir / "dataset_manifest.json", dataset_manifest)

    manifest = {
        "schema_version": "1.0",
        "system": "stratograph",
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack.name,
        "seed": config.seed,
        "benchmarks": manifest_benchmarks,
        "budget": {
            "evaluation_count": budget_meta.get("evaluation_count", len(result_records)),
            "epochs_per_candidate": config.training.epochs,
            "effective_training_epochs": budget_meta.get("effective_training_epochs"),
            "generations": config.evolution.generations,
            "population_size": config.evolution.population_size,
            "budget_policy_name": "prototype_equal_budget",
        },
        "device": {
            "device_name": platform.machine(),
            "precision_mode": "float32",
            "framework": "numpy",
            "framework_version": None,
        },
        "artifacts": {
            "config_snapshot": "config.yaml",
            "report_markdown": str(Path(report_path).relative_to(run_dir)),
            "model_summary_json": "model_summary.json",
            "genome_summary_json": "genome_summary.json",
            "dataset_manifest_json": "dataset_manifest.json",
            "raw_database": "metrics.duckdb",
        },
        "search_telemetry": {
            "qd_enabled": bool(budget_meta.get("qd_enabled", False)),
            "effective_training_epochs": budget_meta.get("effective_training_epochs"),
            "novelty_score_mean": budget_meta.get("novelty_score_mean"),
            "novelty_score_max": budget_meta.get("novelty_score_max"),
            "novelty_archive_final_size": budget_meta.get("novelty_archive_final_size"),
            "map_elites_enabled": bool(budget_meta.get("qd_enabled", False)),
            "map_elites_occupied_niches": budget_meta.get("map_elites_occupied_niches"),
            "map_elites_total_niches": budget_meta.get("map_elites_total_niches"),
            "map_elites_fill_ratio": budget_meta.get("map_elites_fill_ratio"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(result_records, indent=2), encoding="utf-8")
    return manifest_path, results_path


def _write_summary_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
