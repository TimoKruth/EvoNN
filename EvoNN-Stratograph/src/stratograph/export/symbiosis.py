"""Symbiosis-style export contract for Stratograph MLX runs."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evonn_shared.manifests import benchmark_signature, fairness_manifest, summary_core_from_results, write_json

from stratograph.benchmarks import get_benchmark
from stratograph.benchmarks.parity import fallback_native_id, load_parity_pack
from stratograph.config import load_config
from stratograph.export.report import (
    load_report_context,
    load_runtime_metadata,
    summarize_failure_patterns,
    write_report,
)
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
    report_context = load_report_context(run_dir)
    _write_summary_json(output_dir / "genome_summary.json", genomes)
    _write_summary_json(output_dir / "model_summary.json", results)
    dataset_manifest = []
    runtime_meta = load_runtime_metadata(budget_meta)

    manifest_benchmarks: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    for entry in pack.benchmarks:
        native_name = _resolve_native_name(entry, available_results=results)
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
            "wall_clock_seconds": budget_meta.get("wall_clock_seconds"),
            "generations": config.evolution.generations,
            "population_size": config.evolution.population_size,
            "budget_policy_name": "prototype_equal_budget",
        },
        "device": {
            "device_name": platform.machine(),
            "precision_mode": runtime_meta["precision_mode"],
            "framework": runtime_meta["runtime_backend"],
            "framework_version": runtime_meta["runtime_version"],
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
            "architecture_mode": budget_meta.get("architecture_mode", config.evolution.architecture_mode),
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
        "fairness": fairness_manifest(
            pack_name=pack.name,
            seed=config.seed,
            evaluation_count=budget_meta.get("evaluation_count", len(result_records)),
            budget_policy_name="prototype_equal_budget",
            benchmark_entries=manifest_benchmarks,
            data_signature=benchmark_signature(pack.name, manifest_benchmarks),
            code_version=_code_version(),
        ),
    }
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    write_json(manifest_path, manifest)
    write_json(results_path, result_records)
    _write_contract_summary_json(
        output_dir,
        manifest,
        result_records,
        config=config,
        report_context=report_context,
    )
    return manifest_path, results_path


def _resolve_native_name(entry, *, available_results: dict[str, dict[str, Any]]) -> str:
    native_ids = entry.native_ids or {}
    candidates: list[str] = []
    for candidate in [
        native_ids.get("stratograph"),
        fallback_native_id(entry),
        *native_ids.values(),
        entry.benchmark_id,
    ]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate in available_results:
            return candidate
    for candidate in candidates:
        try:
            get_benchmark(candidate)
            return candidate
        except Exception:
            continue
    return candidates[0]


def _write_summary_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _write_contract_summary_json(
    output_dir: Path,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    config,
    report_context: dict[str, Any],
) -> None:
    """Write cross-system summary.json for Stratograph exports."""
    core = summary_core_from_results(
        results=results,
        parameter_counts=[
            int(record["parameter_count"])
            for record in results
            if record.get("status") == "ok" and record.get("parameter_count") is not None
        ],
    )
    budget = manifest.get("budget", {})
    device = manifest.get("device", {})
    runtime_meta = load_runtime_metadata(report_context.get("budget_meta", {}))
    status_payload = report_context.get("status", {})
    representative_genome = report_context.get("representative_genome")
    non_ok_results = report_context.get("non_ok_results", [])

    summary: dict[str, Any] = {
        "system": "stratograph",
        "run_id": manifest["run_id"],
        "status": status_payload.get("state", "complete"),
        "total_evaluations": budget.get("evaluation_count", 0),
        "generations_completed": budget.get("generations", config.evolution.generations),
        "epochs_per_candidate": budget.get("epochs_per_candidate", config.training.epochs),
        "population_size": budget.get("population_size", config.evolution.population_size),
        "wall_clock_seconds": budget.get("wall_clock_seconds"),
        "architecture_mode": manifest.get("search_telemetry", {}).get("architecture_mode")
        or config.evolution.architecture_mode,
        "runtime_backend": device.get("framework", "unknown"),
        "requested_runtime_backend": runtime_meta["requested_runtime_backend"],
        "runtime_version": device.get("framework_version") or "unknown",
        "precision_mode": device.get("precision_mode", "unknown"),
        "runtime_backend_limitations": runtime_meta["runtime_backend_limitations"] or None,
        **core,
        "failure_patterns": dict(summarize_failure_patterns(non_ok_results)),
        "completed_benchmarks": status_payload.get("completed_count", len(results)),
        "remaining_benchmarks": status_payload.get("remaining_count", 0),
        "novelty_mean": manifest.get("search_telemetry", {}).get("novelty_score_mean"),
        "occupied_niches": manifest.get("search_telemetry", {}).get("map_elites_occupied_niches"),
    }
    if representative_genome is not None:
        summary["hierarchy_summary"] = {
            "representative_genome_id": representative_genome.genome_id,
            "macro_nodes": len(representative_genome.macro_nodes),
            "enabled_macro_edges": sum(1 for edge in representative_genome.macro_edges if edge.enabled),
            "cell_library_size": len(representative_genome.cell_library),
            "macro_depth": representative_genome.macro_depth,
            "avg_cell_depth": float(representative_genome.average_cell_depth),
            "reuse_ratio": float(representative_genome.reuse_ratio),
        }
    write_json(output_dir / "summary.json", summary)


def _code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return None
