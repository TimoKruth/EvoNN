"""Symbiosis-style export contract for Primordia runs."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evonn_primordia.benchmarks import get_benchmark
from evonn_primordia.benchmarks.parity import fallback_native_id, load_parity_pack
from evonn_primordia import __version__ as PRIMORDIA_VERSION

from evonn_primordia.config import load_config
from evonn_primordia.export.report import build_primitive_bank_summary, write_report
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
    by_name = {record["benchmark_name"]: record for record in best_results}

    manifest_benchmarks: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    dataset_manifest: list[dict[str, Any]] = []
    for entry in pack.benchmarks:
        native_name = fallback_native_id(entry, "primordia")
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
    runtime_backend = summary.get("runtime", "mlx")
    runtime_version = summary.get("runtime_version")
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
            "generations": 1,
            "population_size": evaluation_count,
            "budget_policy_name": BUDGET_POLICY_NAME,
        },
        "device": {
            "device_name": platform.machine(),
            "precision_mode": "float32",
            "framework": runtime_backend,
            "framework_version": runtime_version,
        },
        "artifacts": {
            "config_snapshot": "config.yaml",
            "report_markdown": "report.md",
            "model_summary_json": "primitive_summary.json",
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
        "fairness": _fairness_manifest(
            pack_name=pack.name,
            seed=config.seed,
            evaluation_count=evaluation_count,
            budget_policy_name=BUDGET_POLICY_NAME,
            benchmark_entries=manifest_benchmarks,
            data_signature=_benchmark_signature(pack.name, manifest_benchmarks),
        ),
    }
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(result_records, indent=2), encoding="utf-8")
    return manifest_path, results_path


def _write_summary_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fairness_manifest(
    *,
    pack_name: str,
    seed: int,
    evaluation_count: int,
    budget_policy_name: str,
    benchmark_entries: list[dict[str, Any]],
    data_signature: str,
) -> dict[str, Any]:
    return {
        "benchmark_pack_id": pack_name,
        "seed": seed,
        "evaluation_count": evaluation_count,
        "budget_policy_name": budget_policy_name,
        "data_signature": data_signature or _benchmark_signature(pack_name, benchmark_entries),
        "code_version": _code_version(),
    }


def _benchmark_signature(pack_name: str, benchmark_entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"pack_name": pack_name, "benchmarks": benchmark_entries},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except Exception:
        return None
