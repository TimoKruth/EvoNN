"""Motif descriptor coverage helpers for Primordia artifacts."""
from __future__ import annotations

from collections import Counter
from math import log2
from statistics import median
from typing import Any


def motif_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, comparable descriptor for one primitive trial."""

    payload = record.get("genome_payload") or {}
    family = str(record.get("primitive_family") or payload.get("family") or "unknown")
    hidden_layers = _int_list(payload.get("hidden_layers"))
    descriptor = {
        "family": family,
        "benchmark_group": str(record.get("benchmark_group") or "unknown"),
        "depth_bucket": _depth_bucket(hidden_layers),
        "width_bucket": _width_bucket(hidden_layers),
        "parameter_bucket": _parameter_bucket(record.get("parameter_count")),
        "activation": str(payload.get("activation") or _activation_hint(record) or "unknown"),
        "norm_type": str(payload.get("norm_type") or "none"),
        "residual": bool(payload.get("residual", False)),
        "dropout_bucket": _dropout_bucket(payload.get("dropout")),
        "sparsity_bucket": _sparsity_bucket(payload.get("activation_sparsity")),
        "kernel_size": int(payload.get("kernel_size") or 0),
        "embedding_bucket": _embedding_bucket(payload.get("embedding_dim")),
        "heads": int(payload.get("num_heads") or 0),
        "experts": int(payload.get("num_experts") or 0),
        "generation_bucket": _generation_bucket(record.get("generation")),
        "mutation_operator": str(record.get("mutation_operator") or "seed"),
    }
    descriptor["descriptor_key"] = descriptor_key(descriptor)
    return descriptor


def descriptor_key(descriptor: dict[str, Any]) -> str:
    """Stable descriptor-cell key for coverage and transfer artifacts."""

    fields = [
        "family",
        "benchmark_group",
        "depth_bucket",
        "width_bucket",
        "parameter_bucket",
        "activation",
        "norm_type",
        "residual",
        "dropout_bucket",
        "sparsity_bucket",
        "generation_bucket",
    ]
    return "|".join(f"{field}={descriptor.get(field)}" for field in fields)


def build_descriptor_coverage(
    *,
    summary: dict[str, Any],
    trial_records: list[dict[str, Any]],
    best_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize how much of Primordia's motif descriptor space was sampled."""

    winning_ids = {
        str(record.get("genome_id"))
        for record in best_results
        if record.get("status") == "ok" and record.get("genome_id") is not None
    }
    cell_records: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    group_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    mutation_counts: Counter[str] = Counter()

    for record in trial_records:
        descriptor = motif_descriptor(record)
        key = str(descriptor["descriptor_key"])
        cell_records.setdefault(key, []).append((descriptor, record))
        group_counts[str(descriptor["benchmark_group"])] += 1
        family_counts[str(descriptor["family"])] += 1
        mutation_counts[str(descriptor["mutation_operator"])] += 1

    descriptor_cells = [
        _descriptor_cell_summary(key, rows, winning_ids=winning_ids)
        for key, rows in cell_records.items()
    ]
    descriptor_cells.sort(
        key=lambda row: (
            -int(row["win_count"]),
            _negative_optional_score(row.get("best_search_score")),
            -int(row["ok_count"]),
            str(row["descriptor_key"]),
        )
    )

    ok_cells = [row for row in descriptor_cells if int(row["ok_count"]) > 0]
    winning_cells = [row for row in descriptor_cells if int(row["win_count"]) > 0]
    group_coverage = _group_coverage(descriptor_cells)
    family_coverage = _family_coverage(descriptor_cells)

    evaluated_count = len(trial_records)
    ok_count = sum(1 for record in trial_records if record.get("status") == "ok")
    return {
        "system": "primordia",
        "run_id": summary.get("run_id"),
        "run_name": summary.get("run_name"),
        "evaluated_motif_count": evaluated_count,
        "successful_motif_count": ok_count,
        "descriptor_cell_count": len(descriptor_cells),
        "successful_descriptor_cell_count": len(ok_cells),
        "winning_descriptor_cell_count": len(winning_cells),
        "descriptor_coverage_ratio": _ratio(len(descriptor_cells), evaluated_count),
        "successful_descriptor_coverage_ratio": _ratio(len(ok_cells), ok_count),
        "descriptor_entropy": _entropy(cell["evaluation_count"] for cell in descriptor_cells),
        "benchmark_group_coverage": group_coverage,
        "family_descriptor_coverage": family_coverage,
        "mutation_operator_counts": dict(sorted(mutation_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "benchmark_group_counts": dict(sorted(group_counts.items())),
        "top_descriptor_cells": descriptor_cells[:12],
    }


def descriptor_support_for_family(
    family: str,
    descriptor_coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return descriptor-support fields scoped to one primitive family."""

    coverage = (descriptor_coverage or {}).get("family_descriptor_coverage") or {}
    row = coverage.get(family, {})
    return {
        "descriptor_count": int(row.get("descriptor_cell_count", 0)),
        "winning_descriptor_count": int(row.get("winning_descriptor_cell_count", 0)),
        "descriptor_coverage_ratio": row.get("descriptor_coverage_ratio"),
        "representative_descriptor": row.get("representative_descriptor"),
        "descriptor_keys": list(row.get("descriptor_keys") or []),
        "transfer_scope": row.get("transfer_scope") or "unknown",
    }


def _descriptor_cell_summary(
    key: str,
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    winning_ids: set[str],
) -> dict[str, Any]:
    descriptor = dict(rows[0][0])
    records = [record for _, record in rows]
    ok_records = [
        record
        for record in records
        if record.get("status") == "ok" and record.get("quality") is not None
    ]
    quality_values = [float(record["quality"]) for record in ok_records]
    search_scores = [
        float(record["search_score"])
        for record in ok_records
        if record.get("search_score") is not None
    ]
    representative = _representative_record(records)
    win_count = sum(1 for record in records if str(record.get("genome_id")) in winning_ids)
    return {
        "descriptor_key": key,
        "descriptor": descriptor,
        "family": descriptor["family"],
        "benchmark_groups": sorted(
            {str(record.get("benchmark_group") or "unknown") for record in records}
        ),
        "evaluation_count": len(records),
        "ok_count": len(ok_records),
        "win_count": win_count,
        "median_quality": median(quality_values) if quality_values else None,
        "best_search_score": max(search_scores) if search_scores else None,
        "representative_genome_id": representative.get("genome_id"),
        "representative_architecture_summary": representative.get("architecture_summary"),
    }


def _group_coverage(descriptor_cells: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for group in sorted({group for cell in descriptor_cells for group in cell["benchmark_groups"]}):
        group_cells = [cell for cell in descriptor_cells if group in cell["benchmark_groups"]]
        families = sorted({str(cell["family"]) for cell in group_cells})
        coverage[group] = {
            "descriptor_cell_count": len(group_cells),
            "winning_descriptor_cell_count": sum(
                1 for cell in group_cells if int(cell["win_count"]) > 0
            ),
            "evaluation_count": sum(int(cell["evaluation_count"]) for cell in group_cells),
            "families": families,
        }
    return coverage


def _family_coverage(descriptor_cells: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for family in sorted({str(cell["family"]) for cell in descriptor_cells}):
        family_cells = [cell for cell in descriptor_cells if str(cell["family"]) == family]
        evaluation_count = sum(int(cell["evaluation_count"]) for cell in family_cells)
        groups = sorted({group for cell in family_cells for group in cell["benchmark_groups"]})
        representative = family_cells[0] if family_cells else {}
        coverage[family] = {
            "descriptor_cell_count": len(family_cells),
            "winning_descriptor_cell_count": sum(
                1 for cell in family_cells if int(cell["win_count"]) > 0
            ),
            "descriptor_coverage_ratio": _ratio(len(family_cells), evaluation_count),
            "descriptor_keys": [str(cell["descriptor_key"]) for cell in family_cells],
            "benchmark_groups": groups,
            "transfer_scope": "multi_group" if len(groups) > 1 else "single_group",
            "representative_descriptor": representative.get("descriptor"),
        }
    return coverage


def _representative_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        records,
        key=lambda record: (
            1 if record.get("status") == "ok" else 0,
            float(record.get("search_score"))
            if record.get("search_score") is not None
            else float("-inf"),
            float(record.get("quality"))
            if record.get("quality") is not None
            else float("-inf"),
        ),
    )


def _entropy(counts: Any) -> float:
    values = [int(count) for count in counts if int(count) > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((count / total) * log2(count / total) for count in values)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _negative_optional_score(value: Any) -> float:
    if value is None:
        return float("inf")
    return -float(value)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    layers: list[int] = []
    for item in value:
        try:
            layers.append(int(item))
        except (TypeError, ValueError):
            continue
    return layers


def _activation_hint(record: dict[str, Any]) -> str | None:
    architecture = str(record.get("architecture_summary") or "")
    if "gelu" in architecture:
        return "gelu"
    if "tanh" in architecture:
        return "tanh"
    if "relu" in architecture:
        return "relu"
    return None


def _depth_bucket(hidden_layers: list[int]) -> str:
    depth = len(hidden_layers)
    if depth <= 1:
        return "single"
    if depth <= 3:
        return "shallow"
    return "deep"


def _width_bucket(hidden_layers: list[int]) -> str:
    if not hidden_layers:
        return "unknown"
    width = max(hidden_layers)
    if width <= 64:
        return "narrow"
    if width <= 160:
        return "mid"
    return "wide"


def _parameter_bucket(value: Any) -> str:
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 0:
        return "unknown"
    if count < 10_000:
        return "tiny"
    if count < 100_000:
        return "small"
    if count < 1_000_000:
        return "medium"
    return "large"


def _dropout_bucket(value: Any) -> str:
    dropout = _float(value)
    if dropout <= 0.0:
        return "none"
    if dropout <= 0.1:
        return "low"
    if dropout <= 0.3:
        return "medium"
    return "high"


def _sparsity_bucket(value: Any) -> str:
    sparsity = _float(value)
    if sparsity <= 0.0:
        return "dense"
    if sparsity <= 0.25:
        return "light"
    if sparsity <= 0.5:
        return "moderate"
    return "sparse"


def _embedding_bucket(value: Any) -> str:
    dim = int(_float(value))
    if dim <= 0:
        return "none"
    if dim <= 64:
        return "small"
    if dim <= 160:
        return "medium"
    return "large"


def _generation_bucket(value: Any) -> str:
    try:
        generation = int(value or 0)
    except (TypeError, ValueError):
        generation = 0
    if generation <= 0:
        return "seed"
    if generation <= 2:
        return "early"
    return "late"


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
