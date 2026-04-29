"""Shared manifest and fairness helpers for EvoNN exports and ingest."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


def default_artifact(run_dir: Path, *candidates: str) -> str:
    """Return the first existing relative artifact path, or the first candidate."""

    for candidate in candidates:
        if (run_dir / candidate).exists():
            return candidate
    return candidates[0]


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write a JSON artifact using the common export formatting."""

    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def summary_core_from_results(
    *,
    results: list[dict[str, Any]],
    parameter_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Derive the common summary-core fields shared by compare/export surfaces."""

    best_fitness: dict[str, float] = {}
    failure_labels: dict[str, int] = {}

    for record in results:
        status = record.get("status")
        metric_value = record.get("metric_value")
        benchmark_id = record.get("benchmark_id")
        if status == "ok" and benchmark_id and metric_value is not None:
            best_fitness[str(benchmark_id)] = float(metric_value)

        reason = record.get("failure_reason") or (status if status not in {None, "ok"} else None)
        if reason is not None:
            label = str(reason)
            failure_labels[label] = failure_labels.get(label, 0) + 1

    sorted_failures = dict(sorted(failure_labels.items(), key=lambda item: (-item[1], item[0])))
    params = list(parameter_counts or [])
    params.sort()
    median_param_count = params[len(params) // 2] if params else 0
    if params and len(params) % 2 == 0:
        median_param_count = int((params[len(params) // 2 - 1] + params[len(params) // 2]) / 2)

    qualities = sorted(best_fitness.values())
    median_quality = None
    if qualities:
        mid = len(qualities) // 2
        median_quality = float(qualities[mid]) if len(qualities) % 2 else float((qualities[mid - 1] + qualities[mid]) / 2)

    return {
        "best_fitness": best_fitness,
        "median_parameter_count": int(median_param_count),
        "median_benchmark_quality": median_quality,
        "failure_count": sum(1 for record in results if record.get("status") != "ok"),
        "failure_patterns": sorted_failures,
        "benchmarks_evaluated": len(best_fitness),
    }


def benchmark_signature(pack_name: str | None, benchmark_entries: list[dict[str, Any]]) -> str:
    """Stable signature for a benchmark pack plus benchmark contract rows."""

    payload = json.dumps(
        {
            "pack_name": pack_name,
            "benchmarks": [
                {
                    "benchmark_id": entry.get("benchmark_id"),
                    "task_kind": entry.get("task_kind"),
                    "metric_name": entry.get("metric_name"),
                    "metric_direction": entry.get("metric_direction"),
                }
                for entry in benchmark_entries
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fairness_manifest(
    *,
    pack_name: str,
    seed: int,
    evaluation_count: int,
    budget_policy_name: str | None,
    benchmark_entries: list[dict[str, Any]],
    data_signature: str | None = None,
    code_version: str | None = None,
) -> dict[str, Any]:
    """Canonical fairness envelope payload builder."""

    return {
        "benchmark_pack_id": pack_name,
        "seed": seed,
        "evaluation_count": evaluation_count,
        "budget_policy_name": budget_policy_name,
        "data_signature": data_signature or benchmark_signature(pack_name, benchmark_entries),
        "code_version": code_version,
    }


def seeding_manifest(
    *,
    seeding_enabled: bool,
    seeding_ladder: str,
    seed_source_system: str | None = None,
    seed_source_run_id: str | None = None,
    seed_artifact_path: str | None = None,
    seed_target_family: str | None = None,
    seed_selected_family: str | None = None,
    seed_rank: int | None = None,
    seed_overlap_policy: str | None = None,
    representative_genome_id: str | None = None,
    representative_architecture_summary: str | None = None,
) -> dict[str, Any]:
    """Canonical seeding envelope payload builder."""

    return {
        "seeding_enabled": seeding_enabled,
        "seeding_ladder": seeding_ladder,
        "seed_source_system": seed_source_system,
        "seed_source_run_id": seed_source_run_id,
        "seed_artifact_path": seed_artifact_path,
        "seed_target_family": seed_target_family,
        "seed_selected_family": seed_selected_family,
        "seed_rank": seed_rank,
        "seed_overlap_policy": seed_overlap_policy,
        "representative_genome_id": representative_genome_id,
        "representative_architecture_summary": representative_architecture_summary,
    }


def _optional_string(value: Any) -> str | None:
    """Normalize optional manifest strings without inventing placeholder values."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def legacy_topograph_primordia_seeding_manifest(payload: Any) -> dict[str, Any] | None:
    """Normalize legacy Topograph seeding metadata without inventing provenance."""

    if not isinstance(payload, Mapping):
        return None
    seed_artifact_path = _optional_string(
        payload.get("seed_artifact_path") or payload.get("seed_path") or payload.get("seed_source_path")
    )
    if seed_artifact_path is None:
        return None
    rank = payload.get("seed_rank", payload.get("selected_rank"))
    return seeding_manifest(
        seeding_enabled=True,
        seeding_ladder="direct",
        seed_source_system=_optional_string(payload.get("seed_source_system") or payload.get("source_system")) or "primordia",
        seed_source_run_id=_optional_string(payload.get("seed_source_run_id")),
        seed_artifact_path=seed_artifact_path,
        seed_target_family=_optional_string(payload.get("seed_target_family", payload.get("target_family"))),
        seed_selected_family=_optional_string(payload.get("seed_selected_family", payload.get("selected_family"))),
        seed_rank=None if rank is None else int(rank),
        seed_overlap_policy=_optional_string(payload.get("seed_overlap_policy")) or "family-overlapping",
        representative_genome_id=_optional_string(payload.get("representative_genome_id")),
        representative_architecture_summary=_optional_string(payload.get("representative_architecture_summary")),
    )


def default_data_signature(payload: dict[str, Any]) -> str:
    """Best-effort signature derivation for legacy manifests lacking fairness metadata."""

    artifacts = payload.get("artifacts", {}) or {}
    dataset_hash = artifacts.get("dataset_manifest_hash")
    if dataset_hash:
        return str(dataset_hash)
    return benchmark_signature(payload.get("pack_name"), payload.get("benchmarks", []))
