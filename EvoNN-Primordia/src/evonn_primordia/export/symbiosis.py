"""Symbiosis-style export contract for Primordia runs."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median as stat_median
from typing import Any

from evonn_shared.manifests import benchmark_signature, fairness_manifest

from evonn_primordia.benchmarks import get_benchmark
from evonn_primordia.benchmarks.parity import fallback_native_id, load_parity_pack
from evonn_primordia import __version__ as PRIMORDIA_VERSION

from evonn_primordia.config import load_config
from evonn_primordia.export.report import build_primitive_bank_summary, load_runtime_metadata, write_report
from evonn_primordia.export.seeding import build_seed_candidates

BUDGET_POLICY_NAME = "prototype_equal_budget"


def export_symbiosis_contract(
    run_dir: str | Path,
    pack_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Export Primordia manifest/results contract for compare tooling."""

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(run_dir / "config.yaml")
    pack = load_parity_pack(pack_path)
    report_path = write_report(run_dir)
    export_config_path = output_dir / "config.yaml"
    export_report_path = output_dir / "report.md"
    if run_dir / "config.yaml" != export_config_path:
        shutil.copy2(run_dir / "config.yaml", export_config_path)
    if Path(report_path) != export_report_path:
        shutil.copy2(report_path, export_report_path)
    best_results = json.loads((run_dir / "best_results.json").read_text(encoding="utf-8"))
    trial_records = json.loads((run_dir / "trial_records.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    runtime_meta = load_runtime_metadata(summary)
    by_name = {record["benchmark_name"]: record for record in best_results}

    manifest_benchmarks: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    dataset_manifest: list[dict[str, Any]] = []
    for entry in pack.benchmarks:
        native_name = _resolve_native_name(entry, available_results=by_name)
        spec = get_benchmark(native_name)
        result = by_name.get(native_name)
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
                "system": "primordia",
                "run_id": summary["run_id"],
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
                "failure_reason": result["failure_reason"] if result else "missing_result",
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

    _write_summary_json(output_dir / "primitive_trials.json", trial_records)
    _write_summary_json(output_dir / "primitive_summary.json", summary)
    _write_summary_json(output_dir / "dataset_manifest.json", dataset_manifest)
    primitive_bank_summary = build_primitive_bank_summary(
        summary=summary,
        best_results=best_results,
        trial_records=trial_records,
    )
    _write_summary_json(output_dir / "primitive_bank_summary.json", primitive_bank_summary)
    seed_candidates = build_seed_candidates(
        summary=summary,
        best_results=best_results,
        trial_records=trial_records,
        primitive_bank=primitive_bank_summary,
    )
    _write_summary_json(output_dir / "seed_candidates.json", seed_candidates)

    evaluation_count = int(summary.get("evaluation_count", len(trial_records)))
    compare_summary = _build_compare_summary(summary=summary, results=result_records, evaluation_count=evaluation_count)
    _write_summary_json(output_dir / "compare_summary.json", compare_summary)
    _write_summary_json(output_dir / "summary.json", compare_summary)

    manifest = {
        "schema_version": "1.0",
        "system": "primordia",
        "version": PRIMORDIA_VERSION,
        "run_id": summary["run_id"],
        "run_name": summary["run_name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pack_name": pack.name,
        "seed": config.seed,
        "benchmarks": manifest_benchmarks,
        "budget": {
            "evaluation_count": evaluation_count,
            "epochs_per_candidate": config.training.epochs_per_candidate,
            "effective_training_epochs": config.training.epochs_per_candidate,
            "wall_clock_seconds": summary.get("wall_clock_seconds"),
            "generations": 1,
            "population_size": evaluation_count,
            "budget_policy_name": BUDGET_POLICY_NAME,
        },
        "device": {
            "device_name": platform.machine(),
            "precision_mode": runtime_meta["precision_mode"],
            "framework": runtime_meta["runtime"],
            "framework_version": runtime_meta["runtime_version"],
        },
        "artifacts": {
            "config_snapshot": "config.yaml",
            "report_markdown": "report.md",
            "model_summary_json": "compare_summary.json",
            "genome_summary_json": "primitive_trials.json",
            "dataset_manifest_json": "dataset_manifest.json",
            "primitive_bank_summary_json": "primitive_bank_summary.json",
            "seed_candidates_json": "seed_candidates.json",
        },
        "search_telemetry": {
            "qd_enabled": False,
            "effective_training_epochs": config.training.epochs_per_candidate,
            "primitive_usage": summary.get("primitive_usage", {}),
            "group_counts": summary.get("group_counts", {}),
            "failure_count": int(summary.get("failure_count", 0)),
            "wall_clock_seconds": summary.get("wall_clock_seconds"),
        },
        "fairness": fairness_manifest(
            pack_name=pack.name,
            seed=config.seed,
            evaluation_count=evaluation_count,
            budget_policy_name=BUDGET_POLICY_NAME,
            benchmark_entries=manifest_benchmarks,
            data_signature=benchmark_signature(pack.name, manifest_benchmarks),
            code_version=_code_version(),
        ),
    }
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(result_records, indent=2), encoding="utf-8")
    return manifest_path, results_path


def _resolve_native_name(entry, *, available_results: dict[str, dict[str, Any]]) -> str:
    native_ids = entry.native_ids or {}
    candidates: list[str] = []
    for candidate in [
        native_ids.get("primordia"),
        fallback_native_id(entry, "primordia"),
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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_compare_summary(
    *,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    evaluation_count: int,
) -> dict[str, Any]:
    best_fitness: dict[str, float] = {}
    ok_param_counts: list[int] = []
    quality_values: list[float] = []
    non_ok_results = [record for record in results if record.get("status") != "ok"]
    failure_patterns = dict(
        Counter(
            str(record.get("failure_reason") or record.get("status") or "unknown")
            for record in non_ok_results
        ).most_common()
    )
    for record in results:
        if record.get("status") != "ok":
            continue
        benchmark_id = str(record.get("benchmark_id") or "")
        metric_value = record.get("metric_value")
        if benchmark_id and metric_value is not None:
            best_fitness[benchmark_id] = float(metric_value)
        parameter_count = record.get("parameter_count")
        if parameter_count is not None:
            ok_param_counts.append(int(parameter_count))
        quality = record.get("quality")
        if quality is not None:
            quality_values.append(float(quality))

    return {
        "system": "primordia",
        "run_id": summary["run_id"],
        "status": summary.get("status", "complete"),
        "total_evaluations": evaluation_count,
        "generations_completed": 1,
        "epochs_per_candidate": int(summary.get("epochs_per_candidate", 0)),
        "population_size": evaluation_count,
        "runtime_backend": summary.get("runtime", "unknown"),
        "runtime_version": summary.get("runtime_version") or "unknown",
        "precision_mode": summary.get("precision_mode", "unknown"),
        "best_fitness": best_fitness,
        "median_parameter_count": int(stat_median(ok_param_counts)) if ok_param_counts else 0,
        "median_benchmark_quality": float(stat_median(quality_values)) if quality_values else None,
        "failure_count": len(non_ok_results),
        "failure_patterns": failure_patterns,
        "benchmarks_evaluated": len(best_fitness),
        "wall_clock_seconds": summary.get("wall_clock_seconds"),
        "primitive_usage": summary.get("primitive_usage", {}),
        "group_counts": summary.get("group_counts", {}),
    }


def _code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return None
