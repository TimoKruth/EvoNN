"""Objective shaping helpers for Primordia candidate ranking."""
from __future__ import annotations

from math import log1p
from typing import Any


def complexity_penalty(record: dict[str, Any], *, benchmark_group: str | None = None) -> float:
    parameter_count = max(0, int(record.get("parameter_count") or 0))
    architecture = str(record.get("architecture_summary") or "")
    width_hint = architecture.count("x") + 1 if "[" in architecture and "]" in architecture else 1
    group_scale = {
        "tabular": 1.0,
        "synthetic": 1.0,
        "image": 0.75,
        "language_modeling": 0.6,
    }.get(benchmark_group or str(record.get("benchmark_group") or "tabular"), 1.0)
    return group_scale * ((log1p(parameter_count) / 10.0) + (0.03 * width_hint))


def parameter_efficiency_score(record: dict[str, Any]) -> float:
    parameter_count = max(0, int(record.get("parameter_count") or 0))
    return 1.0 / max(1.0, log1p(parameter_count))


def train_time_efficiency_score(record: dict[str, Any]) -> float:
    train_seconds = max(0.0, float(record.get("train_seconds") or 0.0))
    return 1.0 / max(1.0, log1p(train_seconds + 1.0))


def novelty_score(record: dict[str, Any], seen_signatures: set[str]) -> float:
    signature = candidate_signature(record)
    if signature not in seen_signatures:
        return 1.0
    family = str(record.get("primitive_family") or "unknown")
    matching_family = sum(1 for existing in seen_signatures if existing.startswith(f"{family}|"))
    return 1.0 / float(1 + matching_family)


def candidate_signature(record: dict[str, Any]) -> str:
    family = str(record.get("primitive_family") or record.get("primitive_name") or "unknown")
    architecture = str(record.get("architecture_summary") or "unknown")
    return f"{family}|{architecture}"


def search_score(
    record: dict[str, Any],
    *,
    benchmark_group: str | None = None,
    novelty_weight: float = 0.05,
    complexity_penalty_weight: float = 0.02,
    seen_signatures: set[str] | None = None,
    selection_mode: str = "composite",
) -> dict[str, float]:
    quality = float(record.get("quality")) if record.get("quality") is not None else float("-inf")
    if selection_mode == "metric_only":
        novelty = 0.0
        complexity = 0.0
        parameter_efficiency = 0.0
        train_efficiency = 0.0
        score = quality
    else:
        novelty = novelty_score(record, seen_signatures or set())
        complexity = complexity_penalty(record, benchmark_group=benchmark_group)
        parameter_efficiency = parameter_efficiency_score(record)
        train_efficiency = train_time_efficiency_score(record)
        score = quality + (novelty_weight * novelty) + (0.02 * parameter_efficiency) + (0.01 * train_efficiency) - (complexity_penalty_weight * complexity)
    return {
        "normalized_quality": quality,
        "novelty_score": novelty,
        "complexity_penalty": complexity,
        "parameter_efficiency_score": parameter_efficiency,
        "train_time_efficiency_score": train_efficiency,
        "search_score": score,
    }
