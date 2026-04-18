"""Symbiosis-style export contract for contender runs."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.benchmarks.parity import fallback_native_id, load_parity_pack
from evonn_contenders.config import load_config
from evonn_contenders.export.report import write_report
from evonn_contenders.storage import RunStore


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
    contenders = store.load_contenders(run["run_id"])
    budget_meta = store.load_budget_metadata(run["run_id"])
    store.close()

    report_path = write_report(run_dir)
    _write_summary_json(output_dir / "contender_summary.json", contenders)
    _write_summary_json(output_dir / "model_summary.json", results)

    manifest_benchmarks: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    dataset_manifest: list[dict[str, Any]] = []
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
                "system": "contenders",
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
                "genome_id": result["contender_id"] if result else None,
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

    _write_summary_json(output_dir / "dataset_manifest.json", dataset_manifest)

    manifest = {
        "schema_version": "1.0",
        "system": "contenders",
        "version": "0.1.0",
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack.name,
        "seed": config.seed,
        "benchmarks": manifest_benchmarks,
        "budget": {
            "evaluation_count": budget_meta.get("evaluation_count", len(contenders)),
            "epochs_per_candidate": 1,
            "effective_training_epochs": 1,
            "generations": 1,
            "population_size": budget_meta.get("evaluation_count", len(contenders)),
            "budget_policy_name": _export_budget_policy_name(budget_meta.get("budget_policy_name")),
        },
        "device": {
            "device_name": platform.machine(),
            "precision_mode": "float32",
            "framework": "scikit-learn",
            "framework_version": None,
        },
        "artifacts": {
            "config_snapshot": "config.yaml",
            "report_markdown": str(Path(report_path).relative_to(run_dir)),
            "model_summary_json": "model_summary.json",
            "contender_summary_json": "contender_summary.json",
            "dataset_manifest_json": "dataset_manifest.json",
            "raw_database": "metrics.duckdb",
        },
        "search_telemetry": {
            "fixed_pool": True,
            "benchmark_count": budget_meta.get("benchmark_count"),
            "tabular_benchmark_count": budget_meta.get("tabular_benchmark_count"),
            "synthetic_benchmark_count": budget_meta.get("synthetic_benchmark_count"),
            "image_benchmark_count": budget_meta.get("image_benchmark_count"),
            "language_modeling_benchmark_count": budget_meta.get("language_modeling_benchmark_count"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(result_records, indent=2), encoding="utf-8")
    return manifest_path, results_path


def _write_summary_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_budget_policy_name(name: Any) -> str:
    if name == "budget_matched_contender_pool":
        return "prototype_equal_budget"
    if name == "fixed_reference_contender_pool":
        return "fixed_reference_contender_pool"
    return str(name or "fixed_reference_contender_pool")
