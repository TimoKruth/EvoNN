"""Contender registry and runtime helpers."""

from evonn_contenders.contenders.registry import (
    ContenderSpec,
    benchmark_group,
    contender_names_for_config,
    resolve_contenders,
)
from evonn_contenders.contenders.runtime import choose_best, evaluate_contender

__all__ = [
    "ContenderSpec",
    "benchmark_group",
    "choose_best",
    "contender_names_for_config",
    "evaluate_contender",
    "resolve_contenders",
]
