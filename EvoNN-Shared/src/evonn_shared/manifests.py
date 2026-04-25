"""Shared manifest and fairness helpers for EvoNN exports and ingest."""

from __future__ import annotations

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


def default_data_signature(payload: dict[str, Any]) -> str:
    """Best-effort signature derivation for legacy manifests lacking fairness metadata."""

    artifacts = payload.get("artifacts", {}) or {}
    dataset_hash = artifacts.get("dataset_manifest_hash")
    if dataset_hash:
        return str(dataset_hash)
    return benchmark_signature(payload.get("pack_name"), payload.get("benchmarks", []))
