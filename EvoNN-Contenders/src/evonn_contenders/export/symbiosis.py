"""Symbiosis-style export contract for contender runs."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evonn_contenders.benchmarks import get_benchmark
from evonn_contenders.benchmarks.parity import fallback_native_id, load_parity_pack
from evonn_contenders.config import load_config
from evonn_contenders.export.report import write_report
from evonn_contenders.storage import RunStore
from evonn_shared.contracts import ArtifactPaths, BenchmarkEntry, BudgetEnvelope, DeviceInfo, ResultRecord, RunManifest
from evonn_shared.manifests import benchmark_signature, fairness_manifest, write_json


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

    manifest_benchmarks: list[BenchmarkEntry] = []
    result_records: list[ResultRecord] = []
    dataset_manifest: list[dict[str, Any]] = []
    for entry in pack.benchmarks:
        native_name = _resolve_native_name(entry, available_results=results)
        spec = get_benchmark(native_name)
        result = results.get(native_name)
        status = result["status"] if result else "missing"
        manifest_benchmarks.append(
            BenchmarkEntry(
                benchmark_id=entry.benchmark_id,
                task_kind=entry.task_kind,
                metric_name=entry.metric_name,
                metric_direction=entry.metric_direction,
                status=status,
            )
        )
        result_records.append(
            ResultRecord(
                system="contenders",
                run_id=run["run_id"],
                benchmark_id=entry.benchmark_id,
                metric_name=entry.metric_name,
                metric_direction=entry.metric_direction,
                metric_value=result["metric_value"] if result else None,
                quality=result["quality"] if result else None,
                parameter_count=result["parameter_count"] if result else None,
                train_seconds=result["train_seconds"] if result else None,
                peak_memory_mb=None,
                architecture_summary=result["architecture_summary"] if result else None,
                genome_id=result["contender_id"] if result else None,
                status=status,
                failure_reason=result["failure_reason"] if result else ("missing_result" if not result else None),
            )
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

    manifest = RunManifest(
        schema_version="1.0",
        system="contenders",
        run_id=run["run_id"],
        run_name=run["run_name"],
        created_at=datetime.now(timezone.utc),
        pack_name=pack.name,
        seed=config.seed,
        benchmarks=manifest_benchmarks,
        budget=BudgetEnvelope(
            evaluation_count=budget_meta.get("evaluation_count", len(contenders)),
            epochs_per_candidate=1,
            effective_training_epochs=1,
            generations=1,
            population_size=budget_meta.get("evaluation_count", len(contenders)),
            budget_policy_name=_export_budget_policy_name(budget_meta.get("budget_policy_name")),
        ),
        device=DeviceInfo(
            device_name=platform.machine(),
            precision_mode="float32",
            framework="scikit-learn",
            framework_version=None,
        ),
        artifacts=ArtifactPaths(
            config_snapshot="config.yaml",
            report_markdown=str(Path(report_path).relative_to(run_dir)),
            model_summary_json="model_summary.json",
            contender_summary_json="contender_summary.json",
            dataset_manifest_json="dataset_manifest.json",
            raw_database="metrics.duckdb",
        ),
        fairness=fairness_manifest(
            pack_name=pack.name,
            seed=config.seed,
            evaluation_count=budget_meta.get("evaluation_count", len(contenders)),
            budget_policy_name=_export_budget_policy_name(budget_meta.get("budget_policy_name")),
            benchmark_entries=[entry.model_dump(mode="json") for entry in manifest_benchmarks],
            data_signature=benchmark_signature(
                pack.name,
                [entry.model_dump(mode="json") for entry in manifest_benchmarks],
            ),
            code_version=_code_version(),
        ),
    )
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    write_json(results_path, [record.model_dump(mode="json") for record in result_records])
    write_json(output_dir / "summary.json", _build_contract_summary(run=run, manifest=manifest, results=result_records))
    return manifest_path, results_path


def _resolve_native_name(entry, *, available_results: dict[str, dict[str, Any]]) -> str:
    native_ids = entry.native_ids or {}
    candidates: list[str] = []
    for candidate in [
        native_ids.get("contenders"),
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


def _build_contract_summary(*, run: dict[str, Any], manifest: RunManifest, results: list[ResultRecord]) -> dict[str, Any]:
    successful = [record for record in results if record.status == "ok"]
    metric_values = [float(record.metric_value) for record in successful if record.metric_value is not None]
    parameter_counts = [int(record.parameter_count) for record in successful if record.parameter_count is not None]
    qualities = [float(record.quality) for record in successful if record.quality is not None]
    failed = [record for record in results if record.status != "ok"]
    failure_patterns: dict[str, int] = {}
    for record in failed:
        label = record.failure_reason or record.status or "unknown"
        failure_patterns[label] = failure_patterns.get(label, 0) + 1
    return {
        "system": "contenders",
        "run_id": manifest.run_id,
        "status": "ok" if not failed else "partial",
        "total_evaluations": manifest.budget.evaluation_count,
        "generations_completed": manifest.budget.generations,
        "epochs_per_candidate": manifest.budget.epochs_per_candidate,
        "population_size": manifest.budget.population_size,
        "runtime_backend": manifest.device.framework,
        "runtime_version": manifest.device.framework_version,
        "precision_mode": manifest.device.precision_mode,
        "best_fitness": max(metric_values) if metric_values else None,
        "median_parameter_count": _median_int(parameter_counts),
        "median_benchmark_quality": _median_float(qualities),
        "failure_count": len(failed),
        "failure_patterns": failure_patterns,
        "benchmarks_evaluated": len(results),
    }



def _write_summary_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _median_float(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _median_int(values: list[int]) -> int | None:
    median = _median_float([float(value) for value in values])
    return None if median is None else int(round(median))


def _export_budget_policy_name(name: Any) -> str:
    if name == "budget_matched_contender_pool":
        return "prototype_equal_budget"
    if name == "fixed_reference_contender_pool":
        return "fixed_reference_contender_pool"
    return str(name or "fixed_reference_contender_pool")



def _code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return None
